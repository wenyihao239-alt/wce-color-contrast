import os, sys, csv, json, cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\wce\wce_experiment")

RES = r"C:\wce\wce_experiment\results"
COND = r"C:\wce\wce_experiment\data\cond"
FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)

# 1. condition montage
rows = list(csv.DictReader(open(os.path.join(RES, "manifest.csv"), encoding="utf-8")))
test = [r for r in rows if r["split"] == "test" and int(r["n_boxes"]) >= 3]
r = test[0]
order = ["A", "C", "M1", "M2", "M3", "P0", "P2", "E"]
tiles = []
for c in order:
    im = cv2.imread(os.path.join(COND, c, "test", r["name"] + ".jpg"))
    tiles.append(cv2.resize(im, (288, 288), interpolation=cv2.INTER_AREA))
grid = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0)
cv2.imwrite(os.path.join(FIG, "conditions_montage.png"), grid)
print("montage", r["name"])

# 2. sweep
s = json.load(open(os.path.join(RES, "sweep.json")))
radii = s["radii"]
de = [s["curves"][str(x)]["dE"] for x in radii]
ls = [s["curves"][str(x)]["ls"] for x in radii]
plt.figure(figsize=(5.2, 3.6))
plt.plot(de, ls, "o-", ms=4)
for i, rr in enumerate(radii):
    if rr in (1, 16, 64, 256):
        plt.annotate(str(rr), (de[i], ls[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
plt.xlabel("CIEDE2000 to M3 (mucosa median)")
plt.ylabel("Lesion-surround Michelson contrast")
plt.title("Cutoff-scale sweep (P0)")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "sweep_frontier.png"), dpi=180)
plt.close()

# 3. G3 bars
g = json.load(open(os.path.join(RES, "gate_g3.json")))
conds = ["A", "M1", "M2", "M3", "C"]
vals = [g["overall"][c]["michelson"] for c in conds]
plt.figure(figsize=(4.8, 3.4))
plt.bar(conds, vals, color=["#777", "#4c72b0", "#dd8452", "#55a868", "#c44e52"])
plt.ylabel("Lesion-surround Michelson contrast")
plt.title("Image-level 2x2 factorial (23,033 boxes)")
plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "gate_g3_contrast.png"), dpi=180)
plt.close()

# 4. G2 gradient ratios
g2 = json.load(open(os.path.join(RES, "gate_g2.json")))
plt.figure(figsize=(4.2, 3.2))
plt.bar(["A", "C"], [g2["green_grad_A"], g2["green_grad_C"]], color=["#777", "#c44e52"])
plt.ylabel("Mean |grad G| at lesion ring")
plt.title(f"G/luminance grad ratio: A={g2['grad_ratio_A_GB']:.3f}, C={g2['grad_ratio_C_GB']:.3f}")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "gate_g2_gradient.png"), dpi=180)
plt.close()
print("figures done")
