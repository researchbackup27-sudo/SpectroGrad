"""Reference implementation of the methods in "SpectroGrad: Diagnosing and Regularizing Spectral Collapse and
Gradient Conflict in Fine-Tuned Image Classifiers"."""

import itertools
import math

import numpy as np
import scipy.stats as sp_stats
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SEEDS", "BATCH_SIZE_DEIT", "batch_size", "TRAIN_RATIO", "VAL_RATIO", "CUB_VAL_RATIO", "IMG_SIZE", "BATCH_SIZE", "FROZEN_EPOCHS",
    "MAX_EPOCHS", "PATIENCE", "LR_HEAD", "LR_FINETUNE", "WEIGHT_DECAY", "COSINE_ETA_MIN", "N_SHI",
    "SHI_EIG_FLOOR", "PENALTY_EIG_FLOOR", "ALPHA", "BETA", "BETA_MAX_CLASSES", "BASELINES",
    "L2ER_WEIGHT_DECAY", "L2ER_LR_GRID", "BACKBONES", "CORRUPTIONS", "ECE_BINS", "PROBE_LAMBDAS",
    "INJECTION_SCALE", "N_INJECTED_PAIRS",
    "stratified_split", "cub_split", "train_transform", "eval_transform", "class_balanced_sampler",
    "build_model", "prelogit_features", "classifier", "shi", "shi_standardised", "layer_shi",
    "per_class_head_gradients", "conflict_matrix", "cgcs_global", "cgcs_batches", "set_seed", "symmetric_confusion", "mantel_test",
    "top_pair_enrichment", "class_size_matrix", "partial_mantel_test", "shi_change", "shi_change_from_history",
    "confusable_pairs", "injection_term", "matched_perturbation", "shi_penalty", "shi_penalty_gram", "shi_penalty_class_weighted",
    "shi_penalty_within_class", "class_frequency_weights", "conflict_aware_projection",
    "spectrograd_step", "baseline_loss", "effective_rank_raw", "l2er_step", "fine_tune", "evaluate",
    "corrupt", "robustness", "ece", "nll", "brier", "linear_probe", "wilcoxon_exact", "sign_test",
    "signflip_test", "holm",
]

SEEDS = (42, 99, 2026)
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
CUB_VAL_RATIO = 0.15
IMG_SIZE, BATCH_SIZE = 224, 32
BATCH_SIZE_DEIT = {"octdl": 32, "gastrovision": 16, "cub200": 16}
FROZEN_EPOCHS, MAX_EPOCHS, PATIENCE = 5, 45, 10
LR_HEAD, LR_FINETUNE, WEIGHT_DECAY, COSINE_ETA_MIN = 1e-3, 1e-4, 1e-4, 1e-6
N_SHI = 256
SHI_EIG_FLOOR = 1e-10
PENALTY_EIG_FLOOR = 1e-7
ALPHA, BETA = 0.05, 0.5
BETA_MAX_CLASSES = 50
BASELINES = {"l2": {"weight_decay": 1e-3}, "label_smoothing": {"label_smoothing": 0.1}, "mixup": {"mixup_alpha": 0.2}}
L2ER_WEIGHT_DECAY, L2ER_LR_GRID = 1e-3, (1e-2, 1e-3, 1e-4)
BACKBONES = {
    "convnext_in12k": "convnext_tiny.in12k_ft_in1k", "convnext_in1k": "convnext_tiny.fb_in1k",
    "deit": "deit_small_patch16_224.fb_in1k", "swin": "swin_tiny_patch4_window7_224.ms_in1k",
    "resnet50_tv": "resnet50.tv_in1k", "resnet50_tv2": "resnet50.tv2_in1k",
}
CORRUPTIONS = {
    "gaussian_noise": (0.04, 0.08, 0.12), "gaussian_blur": (1.0, 2.0, 3.0), "contrast": (0.6, 0.4, 0.2),
    "brightness": (1.3, 1.6, 2.0), "jpeg": (30, 15, 7),
}
ECE_BINS, PROBE_LAMBDAS = 15, (1e-4, 1e-3, 1e-2)
INJECTION_SCALE, N_INJECTED_PAIRS = -0.5, 2
_MEAN, _STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def stratified_split(labels, seed, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """Stratified train / validation / test split (OCTDL, GastroVision)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(labels)); labels = np.asarray(labels)
    trval, test = train_test_split(idx, test_size=1.0 - train_ratio - val_ratio, random_state=seed, stratify=labels)
    train, val = train_test_split(trval, test_size=val_ratio / (train_ratio + val_ratio), random_state=seed,
                                  stratify=labels[trval])
    return train, val, test


def cub_split(train_labels, seed, val_ratio=CUB_VAL_RATIO):
    """CUB-200: the official test split is kept; a stratified val_ratio of the official train split (random_state =
    seed) becomes the validation set."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(train_labels))
    return train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=np.asarray(train_labels))


def batch_size(backbone, dataset):
    """Training batch size: BATCH_SIZE for every backbone except DeiT-S, which used BATCH_SIZE_DEIT[dataset]."""
    return BATCH_SIZE_DEIT[dataset] if "deit" in backbone else BATCH_SIZE


def train_transform():
    from torchvision import transforms
    return transforms.Compose([transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.ToTensor(),
                               transforms.Normalize(_MEAN, _STD)])


def eval_transform():
    from torchvision import transforms
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(),
                               transforms.Normalize(_MEAN, _STD)])


def class_balanced_sampler(labels):
    """WeightedRandomSampler with per-image weight 1 / (class count), drawn with replacement, one epoch =
    len(labels) draws."""
    from torch.utils.data import WeightedRandomSampler
    labels = np.asarray(labels); w = 1.0 / (np.bincount(labels) + 1e-6)
    return WeightedRandomSampler(w[labels].tolist(), len(labels), replacement=True)


def build_model(name, n_classes, pretrained=True):
    """timm backbone with a new linear classifier."""
    import timm
    tag = BACKBONES.get(name, name)
    return timm.create_model(tag, pretrained=pretrained, num_classes=n_classes, drop_path_rate=0.0)


def prelogit_features(model, x):
    """Standard pre-logit features: exactly the input of the classifier (pooling and, for ConvNeXt, the head
    normalisation included)."""
    return model.forward_head(model.forward_features(x), pre_logits=True)


def classifier(model):
    lin = model.get_classifier()
    if not isinstance(lin, nn.Linear):
        raise TypeError(f"expected a linear classifier, got {type(lin).__name__}")
    return lin


def _entropy_rank(eigenvalues):
    p = eigenvalues / eigenvalues.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-30))))


def shi(features):
    """Spectral Health Index: the effective rank exp(H(p)) of the eigenvalues of the centred feature covariance,
    p_i = lambda_i / sum(lambda), float64, eigenvalues floored at SHI_EIG_FLOOR."""
    f = torch.as_tensor(features).double().cpu(); c = f - f.mean(0, keepdim=True)
    cov = (c.T @ c) / (f.shape[0] - 1)
    try:
        ev = torch.linalg.eigvalsh(cov)
    except RuntimeError:
        ev = torch.linalg.svdvals(cov)
    ev = ev.clamp(min=SHI_EIG_FLOOR).numpy()
    v = _entropy_rank(ev)
    return 1.0 if math.isnan(v) else v


def shi_standardised(features):
    """SHI of per-channel z-scored features (the correlation matrix)."""
    f = torch.as_tensor(features).double().cpu(); sd = f.std(0, keepdim=True); keep = (sd > 1e-8).flatten()
    return shi((f[:, keep] - f[:, keep].mean(0, keepdim=True)) / sd[:, keep])


@torch.no_grad()
def layer_shi(model, layers, loader, pool, n=N_SHI, device="cpu"):
    """SHI (raw and standardised) of every listed stage/block output and of the pre-logit features."""
    model.eval(); cap = {}
    hooks = [m.register_forward_hook(lambda mod, i, o, k=k: cap.__setitem__(k, o)) for k, m in layers]
    acc = {k: [] for k, _ in layers}; acc["prelogit"] = []; seen = 0
    try:
        for x, _ in loader:
            if seen >= n:
                break
            x = x.to(device); acc["prelogit"].append(prelogit_features(model, x).float().cpu())
            for k, _ in layers:
                acc[k].append(pool(cap[k]).float().cpu())
            seen += x.shape[0]
    finally:
        for h in hooks:
            h.remove()
    feats = {k: torch.cat(v)[:n] for k, v in acc.items()}
    return {k: shi(v) for k, v in feats.items()}, {k: shi_standardised(v) for k, v in feats.items()}


def per_class_head_gradients(features, labels, weight, bias):
    """Gradient of the class-c mean cross-entropy with respect to the classifier (W, b), for every class present,
    in closed form: E = softmax(F W^T + b) - onehot(y), g_c = [E_c^T F_c / n_c, sum(E_c) / n_c]."""
    with torch.no_grad():
        Fd = torch.as_tensor(features).float(); y = torch.as_tensor(labels).long()
        E = torch.softmax(Fd @ weight.T.float() + bias.float(), dim=1)
        E[torch.arange(len(y)), y] -= 1.0
        classes = torch.unique(y).tolist(); G = []
        for c in classes:
            m = y == c; n = m.sum()
            G.append(torch.cat([((E[m].T @ Fd[m]) / n).flatten(), E[m].sum(0) / n]))
    return torch.stack(G), classes


def conflict_matrix(G):
    """Pairwise conflict between per-class gradients: -cos(g_i, g_j)."""
    Gn = G / G.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return -(Gn @ Gn.T).cpu().numpy()


def cgcs_global(G):
    """Global CGCS: the mean over all class pairs of max(0, -cos); at most K / (2 (K - 1)) for K classes present,
    so compare values only for the same class set."""
    K = conflict_matrix(G); iu = np.triu_indices(len(K), 1)
    return float(np.maximum(0.0, K[iu]).mean()) if len(iu[0]) else 0.0


def cgcs_batches(model, loader, n_classes, n_batches=60, device="cpu"):
    """Batch-averaged CGCS estimator: the pairwise cosine matrix of the closed-form per-class head gradients on
    each of the first n_batches batches of `loader` (model in eval mode), averaged entry by entry over the
    batches in which both classes occur."""
    W, b = classifier(model).weight, classifier(model).bias
    S = np.zeros((n_classes, n_classes)); N = np.zeros((n_classes, n_classes)); model.eval()
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            if t >= n_batches:
                break
            G, classes = per_class_head_gradients(prelogit_features(model, x.to(device)), y.to(device), W, b)
            Gn = G / G.norm(dim=1, keepdim=True).clamp(min=1e-12); C = (Gn @ Gn.T).double().cpu().numpy()
            ix = np.asarray(classes, int); S[np.ix_(ix, ix)] += C; N[np.ix_(ix, ix)] += 1
    with np.errstate(invalid="ignore"):
        return S / N


def set_seed(seed):
    """Seed Python, NumPy and PyTorch and put cuDNN in deterministic mode."""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def symmetric_confusion(confusion):
    """Off-diagonal test confusion between classes i and j, in either direction: C + C^T with zero diagonal."""
    S = np.asarray(confusion, float); S = S + S.T; np.fill_diagonal(S, 0.0)
    return S


def _corr(x, y, method):
    if method == "spearman":
        x, y = sp_stats.rankdata(x), sp_stats.rankdata(y)
    return np.corrcoef(x, y)[0, 1]


def mantel_test(A, B, n_perm=5000, seed=0, method="pearson"):
    """Mantel test between two symmetric class-by-class matrices: Pearson (or, with method="spearman", rank)
    correlation over the upper triangle, and a two-sided permutation p-value from relabelling the classes of B
    (rows and columns together), which keeps the dependence between pairs that share a class."""
    A, B = np.asarray(A, float), np.asarray(B, float); k = A.shape[0]; iu = np.triu_indices(k, 1)
    a = A[iu]; ok = np.isfinite(a); r0 = _corr(a[ok], B[iu][ok], method)
    rng = np.random.default_rng(seed); hits = 0
    for _ in range(n_perm):
        p = rng.permutation(k)
        hits += abs(_corr(a[ok], B[np.ix_(p, p)][iu][ok], method)) >= abs(r0)
    return float(r0), float((hits + 1) / (n_perm + 1)), int(ok.sum())


def top_pair_enrichment(conflict, confusion, top=0.10):
    """Share of all test confusions that fall in the top `top` fraction of class pairs ranked by conflict (`top` is
    expected by chance)."""
    K, S = np.asarray(conflict, float), symmetric_confusion(confusion); iu = np.triu_indices(len(K), 1)
    x, y = K[iu], S[iu]; k = max(1, int(round(top * len(x))))
    return float(y[np.argsort(-x)[:k]].sum() / y.sum()) if y.sum() > 0 else float("nan")


def class_size_matrix(train_counts):
    """log(n_i * n_j) for every class pair: the class-size covariate of partial_mantel_test."""
    n = np.asarray(train_counts, float); return np.log(np.outer(n, n))


def _residual(y, z):
    z = z.reshape(len(y), -1); X = np.column_stack([np.ones(len(y)), z]); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_mantel_test(A, B, Z, n_perm=2000, seed=0, scheme="reresidualise"):
    """Partial Mantel test of A (conflict) against B (symmetric confusion) controlling for Z, one covariate matrix or
    a list of them; one-sided p for a positive association. scheme="reresidualise" permutes A and re-residualises;
    scheme="residual" permutes the residuals of A (Smouse, Long and Sokal, 1986). Returns (r, p)."""
    A, B = np.asarray(A, float), np.asarray(B, float); k = A.shape[0]; iu = np.triu_indices(k, 1)
    Zs = [Z] if np.ndim(Z) == 2 else list(Z)
    z = np.column_stack([np.asarray(m, float)[iu] for m in Zs])
    rb = _residual(B[iu], z); ra = _residual(A[iu], z); r0 = np.corrcoef(ra, rb)[0, 1]
    RA = np.zeros((k, k)); RA[iu] = ra; RA = RA + RA.T
    rng = np.random.default_rng(seed); hits = 0
    for _ in range(n_perm):
        p = rng.permutation(k)
        x = RA[np.ix_(p, p)][iu] if scheme == "residual" else _residual(A[np.ix_(p, p)][iu], z)
        hits += np.corrcoef(x, rb)[0, 1] >= r0
    return float(r0), float((hits + 1) / (n_perm + 1))


def shi_change(shi_start, shi_best):
    """SHI change during fine-tuning, in percent: 100 * (SHI_best - SHI_start) / SHI_start."""
    return 100.0 * (shi_best - shi_start) / shi_start


def shi_change_from_history(history):
    """shi_change from a fine_tune history: SHI after the last head-only epoch -> SHI at the epoch with the lowest
    validation loss."""
    head = [h for h in history if h["phase"] == "head"]; rest = [h for h in history if h["phase"] != "head"]
    start, best = head[-1]["shi"], min(rest, key=lambda h: h["val_loss"])["shi"]
    return start, best, shi_change(start, best)


def confusable_pairs(class_mean_features, k=N_INJECTED_PAIRS):
    """The k class pairs with the highest cosine similarity between class-mean features of a pretrained ImageNet
    network, before any fine-tuning."""
    M = torch.as_tensor(class_mean_features).double(); Mn = M / M.norm(dim=1, keepdim=True).clamp(min=1e-12)
    S = (Mn @ Mn.T).numpy(); iu = np.triu_indices(len(S), 1); order = np.argsort(-S[iu])[:k]
    return [(int(iu[0][t]), int(iu[1][t])) for t in order]


def injection_term(G, classes, pairs, scale=INJECTION_SCALE):
    """Conflict injection: for every injected pair (i, j) with both classes in the batch, add scale * g_j to the
    classifier gradient (scale = -0.5 pushes against class j's own gradient)."""
    pos = {c: t for t, c in enumerate(classes)}; T = None
    for i, j in pairs:
        if i in pos and j in pos:
            T = scale * G[pos[j]] if T is None else T + scale * G[pos[j]]
    return T


def matched_perturbation(term, generator):
    """Control for injection: a random direction with exactly the norm of the injection term, applied on the same
    steps."""
    R = torch.randn(term.numel(), generator=generator).to(term.device, term.dtype)
    return R / (R.norm() + 1e-12) * term.norm()


def shi_penalty(features):
    """Differentiable SHI of a training batch (B, D): SVD of the D x D centred covariance in float64, singular
    values floored at PENALTY_EIG_FLOOR, exp(entropy)."""
    B = features.shape[0]
    if B < 2:
        return features.new_tensor(1.0)
    f = features.double(); c = f - f.mean(0, keepdim=True)
    ev = torch.linalg.svdvals((c.T @ c) / (B - 1)).clamp(min=PENALTY_EIG_FLOOR); p = ev / ev.sum()
    return torch.exp(-(p * torch.log(p + 1e-30)).sum()).to(features.dtype)


def shi_penalty_gram(features):
    """The same value and gradient as shi_penalty, through the B x B Gram matrix: its eigenvalues are the non-zero
    eigenvalues of the D x D covariance, and the remaining D - B eigenvalues are 0 and are floored to the same
    constant."""
    B, D = features.shape
    if B < 2:
        return features.new_tensor(1.0)
    f = features.double(); c = f - f.mean(0, keepdim=True)
    ev = torch.linalg.svdvals((c @ c.T) / (B - 1)) if B < D else torch.linalg.svdvals((c.T @ c) / (B - 1))
    ev = torch.cat([ev, ev.new_zeros(D - ev.numel())]).clamp(min=PENALTY_EIG_FLOOR); p = ev / ev.sum()
    return torch.exp(-(p * torch.log(p + 1e-30)).sum()).to(features.dtype)


def _erank(cov):
    ev = torch.linalg.svdvals(cov).clamp(min=PENALTY_EIG_FLOOR); p = ev / ev.sum()
    return torch.exp(-(p * torch.log(p + 1e-30)).sum())


def class_frequency_weights(labels, class_counts):
    """Per-sample weights proportional to the training-set frequency of each sample's class, summing to one."""
    w = torch.as_tensor(class_counts, dtype=torch.float64)[labels.cpu()]
    return w / w.sum()


def shi_penalty_class_weighted(features, weights):
    """Effective rank of the reliability-weighted batch covariance (weights >= 0, sum 1); uniform weights give shi_penalty."""
    f64 = features.double(); w = weights.double().to(f64.device)
    mu = (w[:, None] * f64).sum(dim=0, keepdim=True); c = (f64 - mu) * w.sqrt()[:, None]
    return _erank((c.T @ c) / (1.0 - (w ** 2).sum())).to(features.dtype)


def shi_penalty_within_class(features, labels):
    """Effective rank of the pooled within-class scatter / (B - K); None when B - K < 2. Its gradient with respect to
    the features has zero mean within each class, so it puts no direct pressure on the class means."""
    classes, idx = torch.unique(labels, return_inverse=True); K = len(classes); dof = features.shape[0] - K
    if dof < 2:
        return None
    f64 = features.double(); oh = F.one_hot(idx, K).double().to(f64.device)
    c = f64 - oh @ ((oh.T @ f64) / oh.sum(dim=0)[:, None])
    return _erank((c.T @ c) / dof).to(features.dtype)


def conflict_aware_projection(G, beta=BETA):
    """Conflict-aware gradient modulation."""
    out = torch.zeros_like(G[0])
    for i in range(G.shape[0]):
        g = G[i].clone()
        for j in range(G.shape[0]):
            if j == i:
                continue
            gj = G[j]; cos = F.cosine_similarity(g.unsqueeze(0), gj.unsqueeze(0))[0]; gjn = gj / (gj.norm() + 1e-10)
            g = torch.where(cos < 0, g - beta * (g @ gjn) * gjn, g)
        out += g
    return out


def spectrograd_step(model, images, labels, optimizer, n_classes, alpha=ALPHA, beta=BETA, penalty="batch",
                     class_counts=None, shuffle_generator=None):
    """One SpectroGrad update; penalty in {"batch", "class_weighted", "shuffled_weight", "within_class"}."""
    model.train(); optimizer.zero_grad()
    feats = prelogit_features(model, images); lin = classifier(model)
    ce = F.cross_entropy(lin(feats), labels)
    if alpha <= 0:
        spec = feats.new_zeros(())
    elif penalty == "batch":
        spec = -alpha * shi_penalty_gram(feats)
    elif penalty in ("class_weighted", "shuffled_weight"):
        w = class_frequency_weights(labels, class_counts)
        if penalty == "shuffled_weight":
            w = w[torch.randperm(len(w), generator=shuffle_generator)]
        spec = -alpha * shi_penalty_class_weighted(feats, w)
    else:
        v = shi_penalty_within_class(feats, labels)
        spec = -alpha * v if v is not None else feats.new_zeros(())
    (ce + spec).backward()
    if beta > 0 and n_classes <= BETA_MAX_CLASSES:
        model.eval()
        with torch.no_grad():
            f_eval = prelogit_features(model, images)
        model.train()
        G, classes = per_class_head_gradients(f_eval, labels, lin.weight, lin.bias)
        combined = conflict_aware_projection(G.to(lin.weight.device), beta)
        nW = lin.weight.numel()
        lin.weight.grad = combined[:nW].reshape(lin.weight.shape).to(lin.weight.dtype)
        lin.bias.grad = combined[nW:].reshape(lin.bias.shape).to(lin.bias.dtype)
    optimizer.step()
    return float(ce.detach()), float(spec.detach())


def baseline_loss(model, images, labels, method):
    """Training loss of the standard regularisers."""
    if method in ("vanilla", "l2"):
        return F.cross_entropy(model(images), labels)
    if method == "label_smoothing":
        return F.cross_entropy(model(images), labels, label_smoothing=BASELINES["label_smoothing"]["label_smoothing"])
    if method == "mixup":
        a = BASELINES["mixup"]["mixup_alpha"]; lam = float(np.random.beta(a, a))
        perm = torch.randperm(images.size(0), device=images.device)
        logits = model(lam * images + (1 - lam) * images[perm])
        return lam * F.cross_entropy(logits, labels) + (1 - lam) * F.cross_entropy(logits, labels[perm])
    raise ValueError(method)


def effective_rank_raw(features):
    """Effective rank used by L2-ER (Prakash et al., 2025): singular values s of the RAW (uncentred) activation
    matrix, p = s / sum(s), exp(H(p)), float64."""
    s = torch.linalg.svdvals(features.double()); p = s / s.sum()
    return torch.exp(-(p * torch.log(p + 1e-12)).sum()).to(features.dtype)


def l2er_step(model, images, er_optimizer, extra_features=None):
    """L2-ER rank step, taken after the ordinary cross-entropy step on the same batch: one plain SGD gradient-
    ASCENT step (er_optimizer = SGD(lr = er_lr)) on the mean effective rank of the pre-logit features and, if
    given, further feature maps."""
    er_optimizer.zero_grad(set_to_none=True)
    feats = [prelogit_features(model, images)] + (list(extra_features(model)) if extra_features else [])
    er = sum(effective_rank_raw(f) for f in feats) / len(feats)
    (-er).backward(); er_optimizer.step()
    return float(er.detach())


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """Logits and labels over a loader (eval mode), plus the mean cross-entropy."""
    model.eval(); L, Y, tot, nb = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device); out = model(x)
        tot += float(F.cross_entropy(out, y)); nb += 1; L.append(out.float().cpu()); Y.append(y.cpu())
    return torch.cat(L).numpy(), torch.cat(Y).numpy(), tot / max(nb, 1)


@torch.no_grad()
def _batch_shi(model, loader, n=N_SHI, device="cpu"):
    model.eval(); out, seen = [], 0
    for x, _ in loader:
        if seen >= n:
            break
        f = prelogit_features(model, x.to(device)).cpu(); out.append(f); seen += f.shape[0]
    return shi(torch.cat(out)[:n])


def fine_tune(model, train_loader, val_loader, n_classes, method="vanilla", device="cpu", er_lr=None,
              extra_features=None, frozen_epochs=FROZEN_EPOCHS, max_epochs=MAX_EPOCHS, patience=PATIENCE,
              alpha=ALPHA, beta=BETA, pairs=None, perturb_seed=0, penalty="batch", class_counts=None):
    """Two-phase fine-tuning used for every method."""
    model.to(device); lin = classifier(model); head = {id(p) for p in lin.parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    hist = []
    for epoch in range(frozen_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        L, Y, vl = evaluate(model, val_loader, device)
        hist.append({"phase": "head", "epoch": epoch, "val_loss": vl, "val_acc": float((L.argmax(1) == Y).mean()),
                     "shi": _batch_shi(model, val_loader, device=device)})
    for p in model.parameters():
        p.requires_grad = True
    if method == "l2er" and not er_lr:
        raise ValueError("method='l2er' needs er_lr (the grid is L2ER_LR_GRID)")
    if method in ("inject", "perturb") and not pairs:
        raise ValueError(f"method='{method}' needs pairs (see confusable_pairs)")
    wd = {"l2": BASELINES["l2"]["weight_decay"], "l2er": L2ER_WEIGHT_DECAY}.get(method, WEIGHT_DECAY)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=COSINE_ETA_MIN)
    er_opt = torch.optim.SGD(model.parameters(), lr=er_lr) if method == "l2er" else None
    gen = torch.Generator().manual_seed(perturb_seed) if method in ("perturb", "spectrograd") else None
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(frozen_epochs, frozen_epochs + max_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if method == "spectrograd":
                spectrograd_step(model, x, y, opt, n_classes, alpha, beta, penalty, class_counts, gen)
            elif method in ("inject", "perturb"):
                model.train(); opt.zero_grad(); feats = prelogit_features(model, x)
                F.cross_entropy(lin(feats), y).backward()
                G, classes = per_class_head_gradients(feats.detach(), y, lin.weight, lin.bias)
                T = injection_term(G.to(lin.weight.device), classes, pairs)
                if T is not None:
                    delta = T if method == "inject" else matched_perturbation(T, gen)
                    nW = lin.weight.numel()
                    lin.weight.grad.add_(delta[:nW].reshape(lin.weight.shape)); lin.bias.grad.add_(delta[nW:])
                opt.step()
            else:
                model.train(); opt.zero_grad()
                baseline_loss(model, x, y, "vanilla" if method == "l2er" else method).backward(); opt.step()
                if er_opt is not None:
                    l2er_step(model, x, er_opt, extra_features)
        sched.step()
        L, Y, vl = evaluate(model, val_loader, device)
        hist.append({"phase": method, "epoch": epoch, "val_loss": vl, "val_acc": float((L.argmax(1) == Y).mean()),
                     "shi": _batch_shi(model, val_loader, device=device)})
        if vl < best:
            best, wait = vl, 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


def corrupt(images, name, severity, noise_seeds=None):
    """Apply one corruption at severity 1, 2 or 3 to uint8 images (N, H, W, 3) that were already resized and
    centre-cropped."""
    from torchvision.io import decode_jpeg, encode_jpeg
    import torchvision.transforms.functional as TF
    level = CORRUPTIONS[name][severity - 1]
    u8 = torch.as_tensor(np.asarray(images, np.uint8)).permute(0, 3, 1, 2).contiguous()
    if name == "jpeg":
        return torch.stack([decode_jpeg(encode_jpeg(im, quality=int(level))) for im in u8]).float() / 255
    x = u8.float() / 255
    if name == "gaussian_noise":
        seeds = noise_seeds if noise_seeds is not None else range(len(x))
        nz = torch.stack([torch.randn(x.shape[1:], generator=torch.Generator().manual_seed(int(s))) for s in seeds])
        return (x + level * nz).clamp(0, 1)
    if name == "gaussian_blur":
        k = 2 * math.ceil(3 * level) + 1
        return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[level, level])
    if name == "contrast":
        return TF.adjust_contrast(x, level)
    if name == "brightness":
        return TF.adjust_brightness(x, level)
    raise ValueError(name)


def robustness(clean_acc, corrupted_accs):
    """mCA = mean accuracy over the 15 corrupted conditions; rCA = mCA / clean accuracy."""
    mca = float(np.mean(corrupted_accs))
    return {"mCA": mca, "rCA": mca / clean_acc if clean_acc > 0 else float("nan")}


def _softmax(z):
    z = np.asarray(z, float); z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(logits, labels, bins=ECE_BINS):
    """Expected calibration error with `bins` equal-width confidence bins, (lo, hi]."""
    p = _softmax(logits); conf, pred = p.max(1), p.argmax(1); cor = (pred == np.asarray(labels)).astype(float); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(cor[m].mean() - conf[m].mean())
    return float(e)


def nll(logits, labels):
    """Mean negative log-likelihood (log loss), probabilities clipped to [eps, 1 - eps] with eps the float64
    machine epsilon and renormalised, as sklearn.metrics.log_loss."""
    eps = np.finfo(np.float64).eps
    p = np.clip(_softmax(logits), eps, 1 - eps); p = p / p.sum(1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(len(labels)), np.asarray(labels)])))


def brier(logits, labels):
    p = _softmax(logits); oh = np.eye(p.shape[1])[np.asarray(labels)]; return float(np.mean(np.sum((p - oh) ** 2, 1)))


def linear_probe(X_train, y_train, X_val, y_val, X_test, y_test, n_classes, lambdas=PROBE_LAMBDAS):
    """Transfer of frozen features: features standardised with the training mean and SD, multinomial logistic
    regression with an L2 penalty lambda * ||W||^2 fitted by L-BFGS for each lambda; the lambda with the best
    validation accuracy is kept (ties go to the larger lambda)."""
    Xtr, Xva, Xte = (torch.as_tensor(np.asarray(a), dtype=torch.float32) for a in (X_train, X_val, X_test))
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    yt = torch.as_tensor(np.asarray(y_train)).long(); best = None
    for lam in lambdas:
        W = torch.zeros(Xtr.shape[1], n_classes, requires_grad=True); b = torch.zeros(n_classes, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=500, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-7, tolerance_change=1e-10)
        def closure():
            opt.zero_grad(); loss = F.cross_entropy(Xtr @ W + b, yt) + lam * (W ** 2).sum(); loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            va = float(((Xva @ W + b).argmax(1).numpy() == np.asarray(y_val)).mean())
            te = float(((Xte @ W + b).argmax(1).numpy() == np.asarray(y_test)).mean())
        if best is None or va > best[0] or (va == best[0] and lam > best[1]):
            best = (va, lam, te)
    return best[2], best[1]


def wilcoxon_exact(d):
    """Two-sided exact Wilcoxon signed-rank test by enumeration of all 2^n sign patterns, with midranks for tied
    |d|."""
    nz = np.asarray(d, float); nz = nz[nz != 0]; n = len(nz)
    if n == 0 or n > 20:
        return float("nan")
    r = sp_stats.rankdata(np.abs(nz)); mu = r.sum() / 2; obs = abs(r[nz > 0].sum() - mu)
    W = np.array(list(itertools.product([0.0, 1.0], repeat=n))) @ r
    return float((np.abs(W - mu) >= obs - 1e-9).mean())


def sign_test(d):
    """Two-sided exact sign test (zeros dropped)."""
    nz = np.asarray(d, float); nz = nz[nz != 0]
    return float(sp_stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else float("nan")


def signflip_test(d):
    """Two-sided exact sign-flip permutation test of the mean paired difference (all 2^n patterns, up to n = 20;
    NaN above)."""
    d = np.asarray(d, float)
    if not 1 <= len(d) <= 20:
        return float("nan")
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float((np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def holm(pvalues):
    """Holm step-down adjustment over the finite p-values; NaN stays NaN."""
    ps = np.asarray(pvalues, float); adj = np.full(len(ps), np.nan); fin = np.where(np.isfinite(ps))[0]; run = 0.0
    for rank, i in enumerate(fin[np.argsort(ps[fin])]):
        run = max(run, min(1.0, (len(fin) - rank) * ps[i])); adj[i] = run
    return adj
