import os, json, time, cv2, numpy as np
import sys
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.color import *

RAW = r"C:\wce\data\raw"
OUT = r"C:\wce\wce_experiment\results"
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
ccm = json.load(open(os.path.join(OUT, "ccm_cccd.json")))
M, b = np.array(ccm["M"]), np.array(ccm["b"])

split, name = "training", sorted(os.listdir(os.path.join(RAW, "origd", "images", "training")))[5]
A = cv2.imread(os.path.join(RAW, "origd", "images", split, name))[:, :, ::-1].astype(np.float64)
C = cv2.imread(os.path.join(RAW, "cccd", "images", split, name))[:, :, ::-1].astype(np.float64)
print("image", name, A.shape)

t0 = time.time()
Crec = np.clip(apply_affine(A, M, b), 0, 255)
Crec8 = np.round(Crec).astype(np.uint8)
labA = rgb_to_lab(A); labC = rgb_to_lab(C); labR = rgb_to_lab(Crec)
dE = delta_e_2000(labC, labR)
print(f"recon vs official: MAE={np.abs(Crec-C).mean():.2f} dE00 median={np.median(dE):.2f} mean={dE.mean():.2f} p95={np.percentile(dE,95):.2f}")
print("time affine+lab", round(time.time()-t0, 3))

Lc, Cc, hc = rgb_to_lch(Crec)
L0, C0, h0 = rgb_to_lch(A)
t0 = time.time()
M1 = lch_to_srgb_gamut(Lc, C0, h0)
M2 = lch_to_srgb_gamut(L0, Cc, hc)
M3 = lch_to_srgb_gamut(Lc, Cc, hc)
print("time 3 gamut maps", round(time.time()-t0, 3))
for tag, im in [("M1", M1), ("M2", M2), ("M3", M3), ("C", C)]:
    print(tag, "meanRGB", im.reshape(-1, 3).mean(0).round(1), "clipR%", round(((im[..., 0] >= 254).mean() * 100), 3),
          "dE_to_C", round(np.median(delta_e_2000(rgb_to_lab(im), labC)), 2))
print("M3 vs Crec identity check", np.abs(M3 - Crec).mean())

# save montage
tiles = [A, C, M1, M2, M3]
labels = ["A Orig", "C CCC-clipped", "M1 Lc", "M2 Cc", "M3 unclipped"]
tiles = [np.clip(np.round(t), 0, 255).astype(np.uint8) for t in tiles]
mont = np.concatenate(tiles, axis=1)
cv2.imwrite(os.path.join(FIG, "condition_sanity.png"), mont[:, :, ::-1])
print("saved", os.path.join(FIG, "condition_sanity.png"))
