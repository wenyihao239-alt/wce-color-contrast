import os, sys, csv, json, argparse, cv2, numpy as np
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.conditions import valid_mask, blend, guided_filter, M_CCC, B_CCC
from wce.color import apply_affine, rgb_to_lch, lch_to_srgb_gamut, rgb_to_lab
from wce.metrics import masks, local_rms, box_ls, de_stats

RES = r"C:\wce\wce_experiment\results"
COND = r"C:\wce\wce_experiment\data\cond"
MANIFEST = os.path.join(RES, "manifest.csv")


def read_rgb(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)


def labels_for(name):
    row = ROW_BY_NAME[name]
    sp = "training" if row["official_split"] == "train" else "validation"
    p = os.path.join(r"C:\wce\data\raw\origd", "labels", sp, name + ".txt")
    if not os.path.exists(p):
        p = os.path.join(r"C:\wce\data\raw\seeai_full", "SEE_AI_project_all_txt", "SEE_AI_project_all_txt", name + ".txt")
    out = []
    if os.path.exists(p):
        for line in open(p):
            q = line.split()
            if len(q) == 5:
                out.append(tuple(float(x) for x in q[1:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    global ROW_BY_NAME
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    ROW_BY_NAME = {r["name"]: r for r in rows}
    val = [r for r in rows if r["split"] == "val" and int(r["n_boxes"]) > 0]
    rng = np.random.default_rng(42)
    if len(val) > args.n:
        val = [val[i] for i in rng.choice(len(val), args.n, replace=False)]
    print("sweep images", len(val), flush=True)
    radii = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    agg = {r: {"dE": [], "rms": [], "ls": [], "ls_w": []} for r in radii}
    # box size distribution
    all_sizes = []
    for k, row in enumerate(val):
        name = row["name"]
        A = read_rgb(os.path.join(COND, "A", "val", name + ".jpg"))
        M3 = read_rgb(os.path.join(COND, "M3", "val", name + ".jpg"))
        m = valid_mask(A)
        mucosa, _, L0, _ = masks(A)
        _, mucosa3, L3, lab3 = masks(M3)
        Crec = np.clip(apply_affine(A, M_CCC, B_CCC), 0, 255)
        Crec = blend(A, Crec, m)
        Lc, Cc, hc = rgb_to_lch(Crec)
        boxes = labels_for(name)
        for b in boxes:
            all_sizes.append(np.sqrt(b[2] * 576 * b[3] * 576))
        for r in radii:
            Gc = guided_filter(Lc, r)
            G0 = guided_filter(L0, r)
            Lp = Gc + (L0 - G0)
            P0 = blend(A, lch_to_srgb_gamut(Lp, Cc, hc), m)
            d = de_stats(P0, lab3, mucosa3 & m)
            agg[r]["dE"].append(d["dE_median"])
            agg[r]["rms"].append(local_rms(rgb_to_lab(P0)[..., 0], mucosa))
            bl = box_ls(rgb_to_lab(P0)[..., 0], m, boxes)
            if len(bl):
                agg[r]["ls"].append(bl[:, 0].mean())
                agg[r]["ls_w"].append(bl[:, 1].mean())
        if k % 25 == 0:
            print("image", k, flush=True)
    out = {"radii": radii, "n_images": len(val), "curves": {}}
    for r in radii:
        out["curves"][r] = {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in agg[r].items()}
    out["box_sqrt_area_px"] = {p: float(np.percentile(all_sizes, p)) for p in [5, 10, 25, 50, 75, 90, 95]} if all_sizes else {}
    json.dump(out, open(os.path.join(RES, "sweep.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
