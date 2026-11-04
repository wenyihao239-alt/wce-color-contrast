import os, sys, csv, json, cv2, numpy as np
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.color import apply_affine, rgb_to_lch, rgb_to_lab, delta_e_2000, lab_to_rgb

RES = r"C:\wce\wce_experiment\results"
V2 = r"C:\wce\wce_experiment\data\v2"
OUT = os.path.join(RES, "v2_audit")
CCM = json.load(open(os.path.join(OUT, "ccm_v2.json")))
M, B = np.array(CCM["M"]), np.array(CCM["b"])
RAW = r"C:\wce\data\raw"


def read(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)


def main():
    rows = list(csv.DictReader(open(os.path.join(RES, "manifest.csv"), encoding="utf-8")))
    val = [r for r in rows if r["split"] == "val"]
    acc = {"n": 0, "dl_x": [], "dl_l_s": [], "dc_l": [], "dc_x": [], "dc_s": [],
           "neg": 0, "pos": 0, "oog": 0, "valid": 0, "de_pub_s": [], "de_pub_o": []}
    for r in val:
        ps = {k: os.path.join(V2, k, "val", r["name"] + ".png") for k in ["O", "L", "X", "S"]}
        if not all(os.path.exists(p) for p in ps.values()):
            continue
        O, L, X, S = [read(ps[k]) for k in ["O", "L", "X", "S"]]
        m = O.max(2) >= 8
        L0, C0, h0 = rgb_to_lch(O)
        Ll, Cl, hl = rgb_to_lch(L)
        Lx, Cx, hx = rgb_to_lch(X)
        Ls, Cs, hs = rgb_to_lch(S)
        acc["dl_x"].append(float(np.median(np.abs(Lx - L0)[m])))
        acc["dl_l_s"].append(float(np.median(np.abs(Ll - Ls)[m])))
        acc["dc_l"].append(float(np.median(np.abs(Cl - C0)[m])))
        acc["dc_x"].append(float(np.median(np.abs(Cx - C0)[m])))
        acc["dc_s"].append(float(np.median(np.abs(Cs - C0)[m])))
        un = apply_affine(O, M, B)
        acc["neg"] += int((un < 0).any(2)[m].sum()); acc["pos"] += int((un > 255).any(2)[m].sum())
        acc["valid"] += int(m.sum())
        lab = np.stack([L0, C0 * np.cos(np.radians(h0)), C0 * np.sin(np.radians(h0))], -1)
        # use unclipped transformed LCh for gamut rate
        Lu, Cu, hu = rgb_to_lch(un)
        tmp = lab_to_rgb(np.stack([Lu, Cu * np.cos(np.radians(hu)), Cu * np.sin(np.radians(hu))], -1), clip=False)
        acc["oog"] += int(((tmp < 0) | (tmp > 255)).any(2)[m].sum())
        # published C vs S/O
        if r["source"] == "origd":
            sp = "training" if r["official_split"] == "train" else "validation"
            p = os.path.join(RAW, "cccd", "images", sp, r["name"] + ".jpg")
            if os.path.exists(p):
                Cpub = read(p)
                acc["de_pub_s"].append(float(np.median(delta_e_2000(rgb_to_lab(Cpub), rgb_to_lab(S))[m])))
                acc["de_pub_o"].append(float(np.median(delta_e_2000(rgb_to_lab(Cpub), rgb_to_lab(O))[m])))
        acc["n"] += 1
        if acc["n"] % 100 == 0:
            print("done", acc["n"], flush=True)
    out = {"n_images": acc["n"],
           "median_abs_dL_X_vs_O": float(np.median(acc["dl_x"])),
           "median_abs_dL_L_vs_S": float(np.median(acc["dl_l_s"])),
           "median_abs_dC_L_vs_O": float(np.median(acc["dc_l"])),
           "median_abs_dC_X_vs_O": float(np.median(acc["dc_x"])),
           "median_abs_dC_S_vs_O": float(np.median(acc["dc_s"])),
           "negative_channel_rate_pct": 100 * acc["neg"] / max(acc["valid"], 1),
           "over255_channel_rate_pct": 100 * acc["pos"] / max(acc["valid"], 1),
           "gamut_oog_rate_pct": 100 * acc["oog"] / max(acc["valid"], 1),
           "median_dE_pub_vs_S": float(np.median(acc["de_pub_s"])) if acc["de_pub_s"] else None,
           "median_dE_pub_vs_O": float(np.median(acc["de_pub_o"])) if acc["de_pub_o"] else None}
    json.dump(out, open(os.path.join(OUT, "e1_component_audit.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
