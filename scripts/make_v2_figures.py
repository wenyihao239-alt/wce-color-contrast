import os, json, cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = r"C:\wce\wce_experiment\results"
FIG = os.path.join(RES, "v2_fig"); os.makedirs(FIG, exist_ok=True)

# montage
order = ["O", "S", "L", "X", "H", "E"]
tiles = []
for c in order:
    im = cv2.imread(os.path.join(FIG, c + ".png"))
    im = cv2.resize(im, (288, 288), interpolation=cv2.INTER_AREA)
    tiles.append(im)
grid = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
cv2.imwrite(os.path.join(FIG, "montage_OSLXHE.png"), grid)

# AP50 best vs last
conds = ["O", "S", "L", "X", "H", "E"]
best, last = [], []
for c in conds:
    p = os.path.join(RES, "v2_eval", f"{c}_s42.json")
    best.append(json.load(open(p))["overall"]["map50"] if os.path.exists(p) else np.nan)
    p2 = os.path.join(RES, "v2_eval", f"{c}_s42_last.json")
    last.append(json.load(open(p2))["overall"]["map50"] if os.path.exists(p2) else np.nan)
x = np.arange(len(conds)); w = 0.38
plt.figure(figsize=(6.2, 3.4))
plt.bar(x - w/2, best, w, label="best.pt")
plt.bar(x + w/2, last, w, label="last.pt")
plt.xticks(x, conds); plt.ylabel("AP50 (low-threshold test)")
plt.ylim(0.40, 0.52); plt.legend(); plt.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "ap50_best_last.png"), dpi=180); plt.close()

# seed variance O/S
pairs = [("O", 42), ("O", 2025), ("S", 42), ("S", 2025)]
vals = []
for c, s in pairs:
    p = os.path.join(RES, "v2_eval", f"{c}_s{s}.json")
    vals.append(json.load(open(p))["overall"]["map50"] if os.path.exists(p) else np.nan)
plt.figure(figsize=(4.2, 3.2))
plt.bar([f"{c}\nseed{s}" for c, s in pairs], vals, color=["#4c72b0", "#4c72b0", "#dd8452", "#dd8452"])
plt.ylabel("AP50"); plt.ylim(0.44, 0.51); plt.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "seed_variance.png"), dpi=180); plt.close()

# E3 grid scatter
p = os.path.join(RES, "v2_audit", "e3_image_grid.json")
if os.path.exists(p):
    d = json.load(open(p))
    plt.figure(figsize=(5.2, 3.6))
    for g in d["grid"]:
        plt.scatter(g["dE_to_S"], g["box_michelson"], c=g["lambda"], cmap="viridis", vmin=0, vmax=1,
                    marker={0: "o", 0.5: "s", 1.0: "^"}[g["eta"]], s=50)
        if g["lambda"] == d["selected"]["lambda"] and g["eta"] == d["selected"]["eta"]:
            plt.scatter(g["dE_to_S"], g["box_michelson"], facecolors="none", edgecolors="red", s=120, linewidths=2)
    plt.xlabel("median ΔE00 to S"); plt.ylabel("bounding-box Michelson contrast")
    plt.title("15-point (λ, η) grid; red = selected H")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "e3_grid.png"), dpi=180); plt.close()

print("figures done")
