# Data and model availability

## Original data

| Dataset | Licence | Link |
|---|---|---|
| SEE-AI full (`kyucapsule`) | CC BY 4.0 | https://www.kaggle.com/datasets/capsuleyolo/kyucapsule |
| R-OrigD YOLO | CC BY 4.0 | https://www.kaggle.com/datasets/emmanuelagosou/see-ai-datatset-r-origd-yolo |
| R-CCD YOLO | CC BY 4.0 | https://www.kaggle.com/datasets/emmanuelagosou/see-ai-datatset-r-ccd-yolo |
| R-CCCD YOLO | CC BY 4.0 | https://www.kaggle.com/datasets/emmanuelagosou/see-ai-datatset-r-cccd-yolo |

The raw data are **not redistributed in this repository**. Derived condition
images and model weights are prepared as Google Drive archives; see
`GDRIVE_UPLOAD.md`.

## Derived conditions

- V1: A, C, M1, M2, M3, P0, P2, E (JPEG; images only; labels in `metadata/labels.zip`)
- V2: O, S, L, X, H, E (PNG; images + labels)

## Attribution

Please cite the SEE-AI project, the WCE2024 baseline and Watine et al. when using
the derived data. The fitted CCC transform is a surrogate rendering, not a
physical calibration of the PillCam SB3.
