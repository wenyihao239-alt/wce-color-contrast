import os, json, time, cv2, numpy as np, sys
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.color import *

RAW = r"C:\wce\data\raw"
OUT = r"C:\wce\wce_experiment\results"
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
ccm = json.load(open(os.path.join(OUT, "ccm_cccd.json")))
M, b = np.array(ccm["M"]), np.array(ccm["b"])


def valid_mask(A, thr=8):
    return A.max(axis=2) >= thr


def blend(A, out, mask):
    res = A.copy()
    res[mask] = out[mask]
    return res

names = sorted(os.listdir(os.path.join(RAW, "origd", "images", "training")))
for idx in [0, 5, 100, 500]:
    name = names[idx]
    A = cv2.imread(os.path.join(RAW, "origd", "images", "training", name))[:, :, ::-1].astype(np.float64)
    C = cv2.imread(os.path.join(RAW, "cccd", "images", "training", name))[:, :, ::-1].astype(np.float64)
    m = valid_mask(A)
    Crec = blend(A, np.clip(apply_affine(A, M, b), 0, 255), m)
    dE = delta_e_2000(rgb_to_lab(C), rgb_to_lab(Crec))
    print(name, "valid%", round(m.mean() * 100, 1), "MAE", round(np.abs(Crec - C).mean(), 2),
          "dE med/mean/p95", np.round(np.percentile(dE, [50]), 2), round(dE.mean(), 2), round(np.percentile(dE, 95), 2),
          "frac>5", round((dE > 5).mean() * 100, 2))

name = names[5]
A = cv2.imread(os.path.join(RAW, "origd", "images", "training", name))[:, :, ::-1].astype(np.float64)
C = cv2.imread(os.path.join(RAW, "cccd", "images", "training", name))[:, :, ::-1].astype(np.float64)
m = valid_mask(A)
Crec = blend(A, np.clip(apply_affine(A, M, b), 0, 255), m)
Lc, Cc, hc = rgb_to_lch(Crec)
L0, C0, h0 = rgb_to_lch(A)
M1 = blend(A, lch_to_srgb_gamut(Lc, C0, h0), m)
M2 = blend(A, lch_to_srgb_gamut(L0, Cc, hc), m)
M3 = blend(A, lch_to_srgb_gamut(Lc, Cc, hc), m)
for tag, im in [("M1", M1), ("M2", M2), ("M3", M3), ("C", C)]:
    print(tag, "meanRGB", im[m].mean(0).round(1), "clipR%", round(((im[..., 0] >= 254) & m).sum() / m.sum() * 100, 3),
          "dE_to_C", round(float(np.median(delta_e_2000(rgb_to_lab(im), rgb_to_lab(C)))), 2))
tiles = [A, C, M1, M2, M3]
tiles = [np.clip(np.round(t), 0, 255).astype(np.uint8) for t in tiles]
cv2.imwrite(os.path.join(FIG, "condition_sanity.png"), np.concatenate(tiles, 1)[:, :, ::-1])
print("saved")
