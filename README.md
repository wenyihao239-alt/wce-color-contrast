# WCE Colour–Contrast Pilot

Reproducible code and evaluation metadata for an exploratory study of colour
correction, lightness/chromatic interventions and pathology detection in wireless
capsule endoscopy (WCE).

> **Status: exploratory.** The experiments do **not** establish a robust method
> ranking or a proven degradation mechanism. In the final V2.1 analysis, H
> (λ=0.25, η=1) obtained AP50 = 0.491 vs S = 0.482, but the difference is within
> run-to-run variation and Holm-corrected tests were non-significant. Input-domain
> shift was much larger than within-domain condition differences.

## Downloads

Derived data, metadata and model weights are available in the shared Google Drive
folder:

**https://drive.google.com/drive/folders/1_BKwtHZlN1nHb1LprpmcSzuJs01Fgguw?usp=sharing**

| Archive | Size | Contents |
|---|---:|---|
| `datasets.zip` | ~36.9 GB | V1 and V2 condition images, 18,481 images per condition |
| `metadata.zip` | ~3.3 MB | labels, manifest, CCM matrices, G1–G3/sweep/E2/E3 JSON, reports |
| `models.zip` | ~402 MB | YOLOv5s best/last weights for V1, V2 and the anchor runs |

The raw SEE-AI images are **not** redistributed in this repository. Download them
from Kaggle (links below) and use the scripts to regenerate conditions if needed.

## What is included

- `src/wce/` — colour-space, condition generation, metrics and data utilities.
- `scripts/` — calibration fitting, condition generation, gates, sweep, training,
  evaluation, statistics and report generation.
- `configs/hyp.controlled.yaml` — controlled YOLOv5 augmentation policy.
- `patches/` — minimal compatibility patch for the public WCE2024/YOLOv5 baseline.
- `metadata/` — manifest, split statistics, CCM matrices, gates, sweep, E2/E3
  results, trimmed evaluation summaries and run manifest.
- `docs/` — full Chinese reports (`WCE_完整实验方案与结果_全整合.md`,
  `WCE_v2.1_评估收尾报告.md`).
- `figures/` — key result figures.

**Not included in git:** raw images, derived condition images (many GB), model
weights. See `DATA.md` for Kaggle links and the Google Drive archive layout.

## Main results (V2.1, `last.pt`, NMS-complete)

| Condition | Epochs | AP50 | AP50-95 | F1@0.24 | FP | Normal FP |
|---|---:|---:|---:|---:|---:|---:|
| O original | 92 | 0.478 | 0.238 | 0.555 | 894 | 65 |
| S CCC rendering | 90 | 0.482 | 0.244 | 0.556 | 851 | 57 |
| L lightness-only | 99 | 0.488 | 0.248 | 0.554 | 914 | 76 |
| X chroma-only | 93 | 0.484 | 0.238 | 0.560 | 948 | 69 |
| H joint control | 97 | 0.491 | 0.245 | 0.563 | 913 | 79 |
| E S+CLAHE | 93 | 0.487 | 0.244 | 0.560 | 865 | 72 |

Frozen-model cross-input AP50 (val):

| Input | f_O | f_S |
|---|---:|---:|
| O | 0.490 | 0.227 |
| S | 0.219 | 0.484 |
| H | 0.255 | 0.458 |

The large cross-domain drop shows that matched-domain retraining is required to
separate image effects from model adaptation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/setup_third_party.sh  # or scripts/setup_third_party.ps1
```

Download `yolov5s.pt` into `third_party/WCE2024/Yolov5/yolov5s.pt`.

## Data

Raw SEE-AI (CC BY 4.0): <https://www.kaggle.com/datasets/capsuleyolo/kyucapsule>

Corrected YOLO datasets (CC BY 4.0):

- <https://www.kaggle.com/datasets/emmanuelagosou/see-ai-datatset-r-origd-yolo>
- <https://www.kaggle.com/datasets/emmanuelagosou/see-ai-datatset-r-ccd-yolo>
- <https://www.kaggle.com/datasets/emmanuelagosou/see-ai-datatset-r-cccd-yolo>

See `DATA.md` for the derived-condition archive layout, sizes and checksums.

## Reproduction sketch

```bash
python scripts/fit_ccm.py
python scripts/fit_ccm_v2.py
python scripts/make_conditions.py --stage base --workers 8
python scripts/make_conditions.py --stage extra --workers 6 --threads
python scripts/sweep.py --n 300
python scripts/choose_params.py
python scripts/prepare_controlled.py --conditions A,C,M1,M2,P0,P2,E
python scripts/run_training.py --conditions A,C,M1,M2,P0,P2,E --seeds 42 --epochs 20 --parallel 2
python scripts/evaluate.py --conditions A,C,M1,M2,P0,P2,E --split test
python scripts/stats.py
```

V2 remote scripts: `make_v2_remote.py`, `make_v2_extra_local.py`, `make_h_remote.py`,
`run_one.sh`, `eval_v2_remote.py`.

## Licence

Original code: MIT (see `LICENSE`). Third-party components and datasets keep their
own licences; see `NOTICE.md` and `DATA.md`.
