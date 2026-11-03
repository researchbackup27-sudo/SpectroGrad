# SpectroGrad

Reference implementation for:

**SpectroGrad: Diagnosing and Regularizing Spectral Collapse and Gradient Conflict in Fine-Tuned Image Classifiers**

## Contents

- `spectrograd.py`: the methods described in the paper: the Spectral Health Index (SHI), its standardised
  and layer-wise forms and the SHI change during fine-tuning; the Class-pair Gradient Conflict Score (CGCS)
  with closed-form per-class classifier gradients (full-split and batch-averaged), its Mantel test against
  class confusion (Pearson or Spearman; partial Mantel with one or more covariates, re-residualised or
  residual permutation) and the top-pair enrichment; the spectral penalty (batch, class-frequency-weighted, within-class, and the shuffled-weight
  control) and the conflict-aware gradient modulation that form SpectroGrad; the baselines (L2, label smoothing, Mixup, L2-ER); the conflict-injection
  ablation with its norm-matched perturbation control; the two-phase fine-tuning protocol; the corruption
  suite and the robustness, calibration and linear-probe measures; seeding; and the exact paired tests
  with Holm adjustment.
- `demo.py`: runs the whole pipeline on synthetic images with a small CNN: vanilla and SpectroGrad
  fine-tuning, CGCS against test confusion, the baselines and the injection ablation, robustness,
  calibration, transfer and layer-wise SHI, followed by implementation checks. Needs none of the image
  datasets, downloads no weights, and finishes in about three minutes on a laptop CPU.

```
pip install -r requirements.txt
python demo.py
```

The backbones in the paper are loaded with `spectrograd.build_model`, which uses the timm weights listed
in `spectrograd.BACKBONES`.

## Datasets

All three are public: OCTDL (Kulyabin et al., 2024), GastroVision (Jha et al., 2023) and CUB-200-2011
(Wah et al., 2011).
