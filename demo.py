"""End-to-end demonstration of the SpectroGrad diagnostics and regulariser on synthetic images."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import spectrograd as sg

N_CLASSES, N_PER_CLASS, SIZE, SEED = 8, 220, 32, 42
FROZEN, MAX_EPOCHS, PATIENCE = 2, 10, 3


class TinyNet(nn.Module):
    """Four convolutional stages, global average pooling and a linear classifier, with the timm interface the
    reference functions use."""
    def __init__(self, n_classes, width=32):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.stages = nn.Sequential(block(3, width), block(width, width), block(width, 2 * width), block(2 * width, 2 * width))
        self.fc = nn.Linear(2 * width, n_classes)

    def forward_features(self, x):
        return self.stages(x)

    def forward_head(self, x, pre_logits=False):
        x = x.mean(dim=(2, 3))
        return x if pre_logits else self.fc(x)

    def get_classifier(self):
        return self.fc

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def make_images(seed, n_classes=N_CLASSES, n_per_class=N_PER_CLASS):
    """Smooth class templates plus noise, as uint8 (N, 32, 32, 3)."""
    rng = np.random.RandomState(seed)
    base = [rng.randn(4, 4, 3) for _ in range(n_classes)]
    base[1] = base[0] + 0.35 * rng.randn(4, 4, 3)
    base[3] = base[2] + 0.35 * rng.randn(4, 4, 3)
    X, y = [], []
    for c in range(n_classes):
        t = np.kron(base[c], np.ones((SIZE // 4, SIZE // 4, 1)))
        for _ in range(n_per_class):
            img = t + 0.9 * rng.randn(SIZE, SIZE, 3) + 0.5 * rng.randn(1, 1, 3)
            X.append(np.clip(127.5 + 40 * img, 0, 255).astype(np.uint8)); y.append(c)
    return np.stack(X), np.array(y)


def to_tensor(u8):
    x = torch.as_tensor(u8).permute(0, 3, 1, 2).float() / 255
    return (x - 0.5) / 0.25


def loaders(X, y, tr, va, te, seed):
    g = torch.Generator().manual_seed(seed)
    ds = lambda idx: TensorDataset(to_tensor(X[idx]), torch.as_tensor(y[idx]).long())
    train = DataLoader(ds(tr), batch_size=sg.BATCH_SIZE, sampler=sg.class_balanced_sampler(y[tr]), drop_last=True,
                       generator=g)
    return train, DataLoader(ds(va), batch_size=64), DataLoader(ds(te), batch_size=64)


def run(method, X, y, split, seed, **kw):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = split; train, val, test = loaders(X, y, tr, va, te, seed)
    model, hist = sg.fine_tune(TinyNet(N_CLASSES), train, val, N_CLASSES, method=method, frozen_epochs=FROZEN,
                               max_epochs=MAX_EPOCHS, patience=PATIENCE, **kw)
    L, Y, _ = sg.evaluate(model, test)
    return model, hist, L, Y


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    X, y = make_images(SEED)
    split = sg.stratified_split(y, SEED)
    tr, va, te = split
    print(f"synthetic data: {len(y)} images, {N_CLASSES} classes, split {len(tr)}/{len(va)}/{len(te)} (stratified 70/15/15)")

    print("\n[1/5] Two-phase fine-tuning (head only, then all weights; early stopping on validation loss)")
    res, hists = {}, {}
    for method in ("vanilla", "spectrograd"):
        model, hist, L, Y = run(method, X, y, split, SEED); hists[method] = hist
        start, best_shi, change = sg.shi_change_from_history(hist)
        best = min(hist[FROZEN:], key=lambda h: h["val_loss"])
        res[method] = (model, L, Y)
        print(f"  {method:12s} test acc {np.mean(L.argmax(1) == Y):.3f} | validation SHI at the start of Phase 2 "
              f"{start:6.2f} -> best epoch {best_shi:6.2f} ({change:+.0f}%) | best epoch {best['epoch']}")
    ch = {m: sg.shi_change_from_history(h)[2] for m, h in hists.items()}
    print(f"  SHI change: SpectroGrad {ch['spectrograd']:+.0f}% vs vanilla {ch['vanilla']:+.0f}%: the penalty "
          + ("raises SHI, as intended" if ch["spectrograd"] > ch["vanilla"] else "did NOT raise SHI on this run"))

    print("\n[2/5] CGCS over the full training split vs test confusion (vanilla model)")
    model, L, Y = res["vanilla"]; model.eval()
    with torch.no_grad():
        F_tr = sg.prelogit_features(model, to_tensor(X[tr]))
    lin = sg.classifier(model)
    G, classes = sg.per_class_head_gradients(F_tr, torch.as_tensor(y[tr]), lin.weight, lin.bias)
    K = sg.conflict_matrix(G)
    from sklearn.metrics import confusion_matrix
    C = confusion_matrix(Y, L.argmax(1), labels=list(range(N_CLASSES)))
    r, p, npairs = sg.mantel_test(K, sg.symmetric_confusion(C), n_perm=2000)
    share = sg.top_pair_enrichment(K, C, top=0.10)
    rp, pp = sg.partial_mantel_test(K, sg.symmetric_confusion(C), sg.class_size_matrix(np.bincount(y[tr])), n_perm=2000)
    iu = np.triu_indices(N_CLASSES, 1); top = np.argsort(-K[iu])[:2]
    print(f"  global CGCS {sg.cgcs_global(G):.4f} | Mantel r = {r:+.3f}, permutation p = {p:.4f} over {npairs} class pairs")
    print(f"  controlling for class size (partial Mantel): r = {rp:+.3f}, one-sided p = {pp:.4f}")
    rs, ps, _ = sg.mantel_test(K, sg.symmetric_confusion(C), n_perm=2000, method="spearman")
    with torch.no_grad():
        Ptr = torch.softmax(lin(F_tr), dim=1).numpy()
    soft = np.stack([Ptr[y[tr] == c].mean(0) for c in range(N_CLASSES)]); soft = soft + soft.T; np.fill_diagonal(soft, 0)
    Fm = np.stack([F_tr[torch.as_tensor(y[tr]) == c].mean(0).numpy() for c in range(N_CLASSES)])
    Fm = Fm / np.linalg.norm(Fm, axis=1, keepdims=True)
    covs = [sg.class_size_matrix(np.bincount(y[tr])), soft, Fm @ Fm.T]
    ra, pa = sg.partial_mantel_test(K, sg.symmetric_confusion(C), covs, n_perm=2000, scheme="residual")
    train_loader = loaders(X, y, tr, va, te, 1000 + SEED)[0]
    Kb = -sg.cgcs_batches(model, train_loader, N_CLASSES, n_batches=60); ok = np.isfinite(Kb[iu])
    print(f"  Spearman Mantel r = {rs:+.3f}, p = {ps:.4f} | controlling for size, soft confusion and class similarity "
          f"(residual permutation): r = {ra:+.3f}, one-sided p = {pa:.4f}")
    print(f"  batch-averaged estimator vs full-split conflict: r = {np.corrcoef(Kb[iu][ok], K[iu][ok])[0, 1]:.2f}")
    print(f"  top-10% most conflicting pairs hold {100 * share:.0f}% of the test confusions (10% by chance)")
    print(f"  most conflicting pairs: {[(int(iu[0][i]), int(iu[1][i])) for i in top]} (built to be confusable: (0, 1), (2, 3))")

    print("\n[3/5] Baselines and the conflict-injection ablation")
    counts = np.bincount(y[tr], minlength=N_CLASSES)
    for method, kw in (("l2", {}), ("label_smoothing", {}), ("mixup", {}), ("l2er", {"er_lr": 1e-4}),
                       ("spectrograd", {"penalty": "class_weighted", "class_counts": counts}),
                       ("spectrograd", {"penalty": "shuffled_weight", "class_counts": counts}),
                       ("spectrograd", {"penalty": "within_class"})):
        _, hist, L, Y = run(method, X, y, split, SEED, **kw)
        best = min(hist[FROZEN:], key=lambda h: h["val_loss"])
        name = kw.get("penalty", method)
        print(f"  {name:16s} test acc {np.mean(L.argmax(1) == Y):.3f} | SHI at best epoch {best['shi']:6.2f}")
    torch.manual_seed(SEED); ref_net = TinyNet(N_CLASSES).eval()
    with torch.no_grad():
        Fr = sg.prelogit_features(ref_net, to_tensor(X[tr]))
    means = torch.stack([Fr[torch.as_tensor(y[tr]) == c].mean(0) for c in range(N_CLASSES)])
    pairs = sg.confusable_pairs(means)
    for method in ("inject", "perturb"):
        _, hist, L, Y = run(method, X, y, split, SEED, pairs=pairs, perturb_seed=SEED)
        best = min(hist[FROZEN:], key=lambda h: h["val_loss"])
        print(f"  {method:16s} pairs {pairs} | test acc {np.mean(L.argmax(1) == Y):.3f} | SHI at best epoch {best['shi']:6.2f}")

    print("\n[4/5] Robustness, calibration, transfer, layer-wise SHI")
    for method in ("vanilla", "spectrograd"):
        model, L, Y = res[method]; model.eval()
        clean = float(np.mean(L.argmax(1) == Y)); accs = []
        with torch.no_grad():
            for name in sg.CORRUPTIONS:
                for sev in (1, 2, 3):
                    xc = (sg.corrupt(X[te], name, sev, noise_seeds=te) - 0.5) / 0.25
                    accs.append(float((model(xc).argmax(1).numpy() == y[te]).mean()))
        rb = sg.robustness(clean, accs)
        print(f"  {method:12s} clean {clean:.3f}, mCA {rb['mCA']:.3f}, rCA {rb['rCA']:.3f} | ECE {sg.ece(L, Y):.3f}, "
              f"NLL {sg.nll(L, Y):.3f}, Brier {sg.brier(L, Y):.3f}")
    X2, y2 = make_images(SEED + 1, n_classes=5, n_per_class=120); a, b, c = sg.stratified_split(y2, SEED)
    model = res["vanilla"][0]
    with torch.no_grad():
        F2 = sg.prelogit_features(model, to_tensor(X2)).numpy()
    acc, lam = sg.linear_probe(F2[a], y2[a], F2[b], y2[b], F2[c], y2[c], 5)
    print(f"  linear probe of the frozen vanilla features on a new 5-class task: test acc {acc:.3f} (lambda {lam:g})")
    layers = [(f"stage{i}", m) for i, m in enumerate(model.stages)]
    _, val, _ = loaders(X, y, tr, va, te, SEED)
    raw, z = sg.layer_shi(model, layers, val, pool=lambda o: o.mean(dim=(2, 3)), n=256)
    print("  layer SHI (raw / standardised): " + ", ".join(f"{k} {raw[k]:.1f}/{z[k]:.1f}" for k in raw))

    print("\n[5/5] Implementation checks")
    torch.manual_seed(0)
    Fx, yx = torch.randn(64, 20), torch.randint(0, 5, (64,)); head = nn.Linear(20, 5)
    G, cl = sg.per_class_head_gradients(Fx, yx, head.weight, head.bias); worst = 0.0
    for k, c in enumerate(cl):
        head.zero_grad(); nn.functional.cross_entropy(head(Fx[yx == c]), yx[yx == c]).backward()
        worst = max(worst, float((torch.cat([head.weight.grad.flatten(), head.bias.grad]) - G[k]).abs().max()))
    print(f"  closed-form per-class gradients vs autograd: max abs diff {worst:.2e}"
          f"{'  IDENTICAL' if worst < 1e-5 else '  *** DIFFERS ***'}")
    T = sg.injection_term(G, cl, [(cl[0], cl[1]), (cl[2], cl[3])])
    ref_T = sg.INJECTION_SCALE * (G[1] + G[3])
    P = sg.matched_perturbation(T, torch.Generator().manual_seed(0))
    print(f"  injection term = -0.5 x (g_j summed over the injected pairs): {bool(torch.allclose(T, ref_T))}; "
          f"matched perturbation has the same norm: {abs(float(P.norm() - T.norm())) < 1e-5}")
    f = torch.randn(32, 512, dtype=torch.float64)
    a_, b_ = f.clone().requires_grad_(True), f.clone().requires_grad_(True)
    va_, vb_ = sg.shi_penalty(a_), sg.shi_penalty_gram(b_); va_.backward(); vb_.backward()
    dv, dg = abs(float(va_ - vb_)) / float(va_), float((a_.grad - b_.grad).abs().max() / a_.grad.abs().max())
    print(f"  spectral penalty, B x B Gram form vs D x D form: value {dv:.1e}, gradient {dg:.1e}"
          f"{'  IDENTICAL' if dv < 1e-10 and dg < 1e-8 else '  *** DIFFERS ***'}")
    small = torch.randn(8, 24, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(sg.shi_penalty, (small,), eps=1e-7, atol=1e-6, rtol=1e-4)
    print(f"  torch.autograd.gradcheck of the penalty: {ok}")
    with torch.no_grad():
        F_te = sg.prelogit_features(res["vanilla"][0], to_tensor(X[va][:256]))
    print(f"  layer_shi hook / loader path: pre-logit entry equals shi() on directly computed features: "
          f"{abs(raw['prelogit'] - sg.shi(F_te)) < 1e-9}")
    from scipy import stats
    d = np.random.RandomState(1).randn(9) + 0.4
    ref = stats.wilcoxon(d, method="exact").pvalue
    print(f"  exact Wilcoxon (enumeration) vs scipy exact: {sg.wilcoxon_exact(d):.6f} vs {ref:.6f}"
          f"{'  IDENTICAL' if abs(sg.wilcoxon_exact(d) - ref) < 1e-12 else '  *** DIFFERS ***'}")
    print(f"  sign test / sign-flip test on the same differences: {sg.sign_test(d):.4f} / {sg.signflip_test(d):.4f}")
    dt = np.array([0.5, -0.5, 1.0, 1.0, -2.0, 0.5, 3.0, 1.0])
    ad = np.abs(dt); ranks = np.array([(np.sum(ad < v) + np.sum(ad <= v) + 1) / 2 for v in ad])
    w_obs = ranks[dt > 0].sum(); mu = ranks.sum() / 2; cnt = 0
    for signs in np.ndindex(*(2,) * len(dt)):
        cnt += abs(ranks[np.array(signs) == 1].sum() - mu) >= abs(w_obs - mu) - 1e-9
    bf = cnt / 2 ** len(dt)
    print(f"  exact Wilcoxon with ties (midranks) vs independent brute force: {sg.wilcoxon_exact(dt):.6f} vs {bf:.6f}"
          f"{'  IDENTICAL' if abs(sg.wilcoxon_exact(dt) - bf) < 1e-12 else '  *** DIFFERS ***'}")
    print(f"  Holm on [0.01, 0.04, 0.03, nan]: {np.round(sg.holm([0.01, 0.04, 0.03, np.nan]), 4).tolist()} "
          f"(expected [0.03, 0.06, 0.06, nan])")


if __name__ == "__main__":
    main()
