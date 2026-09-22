# AI Signature Verification — Experiment V1 (WRITER-INDEPENDENT)

**STATUS: EXPERIMENTAL RESEARCH CODE.**
This package is intentionally **isolated** from the FastAPI production service
(`app/`). Nothing under `ai/` is imported by the production API. The existing
OpenCV baseline (`opencv_baseline / v1`) is **not modified** and is only used as
a comparison reference.

## Objective

Build and objectively evaluate a first **writer-independent** Siamese neural
network (ResNet18 backbone, PyTorch) on the CEDAR signature dataset, using a
**writer-level** split so that no test writer ever appears in training.

## Environment used for this run

- Experiment venv (separate from the FastAPI `.venv`): `C:\Users\nassime khatib\aiv`
- Python 3.13.15, torch **2.8.0+cu128**, torchvision 0.23.0, numpy 2.2.6,
  opencv-python-headless 4.14.0.94, matplotlib, pytest
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop (12.8 GB VRAM, sm_120), CUDA 12.8
- torch 2.13.0 was **unusable**: its freshly-published binaries were blocked on
  this machine by Windows application-control (WinError 4551). Fixed by pinning
  to the well-established **2.8.0+cu128** release.

> NOTE: a local `ai/.venv` is not used because `site-packages` paths exceeded
> Windows MAX_PATH when installing torch (its license tree is deep). The venv
> lives at the short path `C:\Users\nassime khatib\aiv`.

## Dataset (CEDAR)

- Root: `C:\Users\nassime khatib\Downloads\archive\CEDAR\CEDAR`
- 55 writers, folders `1`..`55`, each with exactly:
  - 24 genuine signatures: `original_<writer>_<n>.png`
  - 24 skilled forgeries: `forgeries_<writer>_<n>.png`
- All 2640 images decoded cleanly (verified with OpenCV). Not modified.

## Writer split (deterministic, seed = 42)

Target 70/15/15 across whole writers -> nearest exact split with 55 writers:

| split | count | writers |
|---|---|---|
| train | 38 (69.1%) | 1, 4, 5, 10-14, 17, 19-27, 29-34, 36, 37, 39, 40, 42, 43, 45, 47, 49-52, 54, 55 |
| validation | 8 (14.5%) | 3, 6, 28, 35, 38, 44, 46, 53 |
| test | 9 (16.4%) | 2, 7, 8, 9, 15, 16, 18, 41, 48 |

- Writer **7 is forced into TEST** so the signer-7 secondary benchmark is a true
  unseen-writer evaluation (this is decided **before** training).
- Disjointness is asserted at runtime (`assert`s in `compute_writer_split` and
  `validate_no_writer_leakage`).

## Pair generation

Balanced, lazy (pairs are light tuples; images are read+preprocessed on demand):

| split | positive | skilled forgery | random impostor | total |
|---|---|---|---|---|
| train (pool) | 10 488 | 21 888 | 4 608 | 36 984 |
| validation | 2 208 | 4 608 | 16 128 | 22 944 |
| test | 2 484 | 5 184 | 20 736 | 28 404 |

- positive = 2 distinct genuine signatures of the **same** writer
- skilled forgery = genuine + forgery of the **same claimed** writer
- random impostor = genuine of writer A + genuine of writer B (A != B)
- Each training epoch samples a **balanced** batch (50% positive / 25% skilled /
  25% impostor) deterministically seeded by `(seed, epoch)`.
- Validation and test use the full pools (deterministic, no augmentation).

## Preprocessing (one shared implementation)

`preprocessing.py` — identical for train/eval:
1. decode grayscale; 2. blur 3×3; 3. Otsu + `THRESH_BINARY_INV` (ink=foreground);
4. crop to ink bbox; 5. **aspect-ratio-preserving** resize; 6. center-pad to
   **256×128** canvas (no stretching).

Grayscale is broadcast to **3 channels** (choice documented: keep the pretrained
ImageNet ResNet18 conv1 untouched) and ImageNet-normalized. Ink is encoded as
*high* (1.0), background *low* (0.0) — same polarity as the OpenCV baseline.

## Augmentation (train only)

- rotation ±5°, translation ±4 px, scale 0.95–1.05 (affine, border = background)
- No flips, no perspective warp, no large rotations (writer identity preserved).
- Validation/test: **no augmentation** (deterministic).

## Model

- Siamese, single shared ResNet18 backbone (`encode()` on both inputs)
- ImageNet V1 pretrained weights **verified loaded** (torchvision)
- classification head removed; projection MLP 512 -> 256 -> **128**
- embeddings **L2-normalized**; similarity = cosine, distance = Euclidean
- ~11.34 M parameters

## Loss

Contrastive, margin = 1.0 (configurable). Label semantics:
`y = 1` same-writer/genuine, `y = 0` non-match.

```
L = mean( y * d^2 + (1 - y) * max(margin - d, 0)^2 )   d = ||e1 - e2||_2
```

The verification threshold is **never** part of the loss; it is selected from
validation scores after training.

## Training hyperparameters

| param | value |
|---|---|
| epochs | 20 (early stop patience 6) |
| pairs/epoch | 4000 |
| batch size | 64 |
| optimizer | AdamW |
| lr / weight decay | 1e-4 / 1e-4 |
| AMP | yes (CUDA) |
| model selection | best validation EER |

## Evaluation protocol (no threshold cheating)

- Operational threshold = **validation** EER threshold (never derived from test).
- Test set (unseen writers) is used once, with that threshold.
- Test-side EER/ROC are also reported as descriptive statistics.
- Subgroup breakdown: positives / skilled forgery / random impostor.
- Skilled-forgery FAR is reported separately (a model may ace random impostors
  while failing skilled forgeries).

## Signer #7 secondary benchmark

Writer 7 is in the test split, so it is a valid unseen-writer evaluation.
Protocol mirrors the previous OpenCV baseline: reference `original_7_5.png` vs
the other 23 originals and vs the 24 forgeries. Genuine/forgery score
distributions are reported for comparison.

## Run

```powershell
$py = "C:\Users\nassime khatib\aiv\Scripts\python.exe"
cd services/verification-api
& $py -m ai.train                 # train + evaluate + reports (GPU auto-detected)
& $py -m pytest -c ai/pytest.ini  # unit tests (no training required)
```

Outputs:
- checkpoint: `ai/checkpoints/siamese_resnet18_v1.pt` (full metadata incl. the
  validation-derived threshold; NOT used by FastAPI)
- reports: `ai/reports/{training_history,validation_pairs,test_pairs}.csv`,
  `metrics.json`, `roc_curve.png`, `score_distribution.png`

## Files

```
ai/
  README.md            this file
  config.py            all hyper-parameters + device detection
  dataset.py           CEDAR scan, writer split, pair pools, PairDataset
  preprocessing.py     shared pipeline + train-only augmentation
  model.py             Siamese ResNet18 + projection head
  losses.py            ContrastiveLoss (margin configurable)
  metrics.py           ROC / AUC / FAR / FRR / EER
  train.py             training entry point (python -m ai.train)
  evaluate.py          unseen-writer eval, signer-7, reports/plots
  checkpoints/         siamese_resnet18_v1.pt
  reports/             CSVs, JSON, PNG artifacts
  tests/               pytest unit tests
  pytest.ini
```

## Do-not-touch guarantees

No changes to ASP.NET, SQL/EF Core/migrations, the domain entities, extraction
V2.4/ROI/re-extraction, the OpenCV baseline algorithm, `/api/signatures/compare`,
PaddleOCR/OCR. This experiment will not be wired into FastAPI without a separate
decision.
