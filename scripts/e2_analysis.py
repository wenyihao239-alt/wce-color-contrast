import os, sys, json, csv, cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.color import apply_affine, rgb_to_lch, lch_to_srgb_gamut, rgb_to_lab, lab_to_rgb
from wce.conditions import guided_filter

RES = r"C:\wce\wce_experiment\results"
V2 = r"C:\wce\wce_experiment\data\v2"
OUT = os.path.join(RES, "v2_audit", "e2")
FIG = os.path.join(RES, "v2_audit", "figures"); os.makedirs(FIG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
CCM = json.load(open(os.path.join(RES, "v2_audit", "ccm_v2.json")))
M, B = np.array(CCM["M"]), np.array(CCM["b"])


def michelson(L):
    return float((L.max() - L.min()) / (L.max() + L.min() + 1e-9))


def transform_l(rgb):
    un = apply_affine(rgb, M, B)
    L, C, h = rgb_to_lch(un)
    out = lch_to_srgb_gamut(L, C, h)
    return out, rgb_to_lab(out)[..., 0]


def main():
    out = {}
    # synthetic 1: affine offset changes normalized contrast without blur
    a, b = 0.80, 10.0
    rows = []
    for (m1, m2) in [(30, 60), (40, 50), (50, 70), (20, 80)]:
        K = (m2 - m1) / (m1 + m2)
        Kp = (a * (m2 - m1)) / (a * (m1 + m2) + 2 * b)
        rows.append([m1, m2, K, Kp, Kp / K])
    out["affine_offset_contrast"] = rows

    # synthetic 2: step edge + guided filter
    step = np.zeros((128, 256, 3), float); step[:, :128] = 40; step[:, 128:] = 80
    Lstep = rgb_to_lab(step)[..., 0]
    gf = guided_filter(Lstep, 64)
    out["step_edge"] = {"input_low": float(Lstep[:, 20].mean()), "input_high": float(Lstep[:, -20].mean()),
                        "gf_low": float(gf[:, 20].mean()), "gf_high": float(gf[:, -20].mean()),
                        "gf_center": float(gf[64, 128]), "input_center": float(Lstep[64, 128]),
                        "input_contrast": michelson(Lstep), "gf_contrast": michelson(gf)}

    # synthetic 3: sinusoid amplitude before/after operations
    sinus = []
    for period in [4, 8, 16, 32, 64, 128]:
        x = np.arange(256)
        v = 128 + 20 * np.sin(2 * np.pi * x / period)
        im = np.repeat(v[None, :, None], 128, 0).repeat(3, 2)
        L = rgb_to_lab(im)[..., 0]
        _, Lt = transform_l(im)
        gf = guided_filter(L, 64)
        gauss = cv2.GaussianBlur(L, (0, 0), 64 / 3)
        roundtrip = rgb_to_lab(lab_to_rgb(rgb_to_lab(im)))[..., 0]
        sinus.append({"period": period, "input_std": float(L.std()), "affine_std": float(Lt.std()),
                      "gf_std": float(gf.std()), "gauss_std": float(gauss.std()),
                      "roundtrip_std": float(roundtrip.std())})
    out["sinusoid"] = sinus

    # synthetic 4: gamut mapping rate on a colour ramp
    hsvx = np.zeros((64, 256, 3), np.uint8)
    hsvx[..., 0] = np.tile(np.arange(256, dtype=np.uint8)[None, :], (64, 1))
    hsvx[..., 1] = 255; hsvx[..., 2] = 200
    rgb = cv2.cvtColor(hsvx, cv2.COLOR_HSV2RGB).astype(float)
    un = apply_affine(rgb, M, B)
    L, C, h = rgb_to_lch(un)
    mapped = lch_to_srgb_gamut(L, C, h)
    oog = 0
    from wce.color import lab_to_rgb as l2r
    tmp = l2r(np.stack([L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h))], -1), clip=False)
    oog = int(((tmp < 0) | (tmp > 255)).any(2).mean() * 100)
    out["colour_ramp_gamut_rate_pct"] = oog

    # h(L): fit on available O/S train pairs
    rows_m = list(csv.DictReader(open(os.path.join(RES, "manifest.csv"), encoding="utf-8")))
    tr = [r for r in rows_m if r["split"] == "train"]
    rng = np.random.default_rng(0)
    LOs, LCs, C0s = [], [], []
    used = 0
    for r in tr[::20]:
        po = os.path.join(V2, "O", "train", r["name"] + ".png")
        ps = os.path.join(V2, "S", "train", r["name"] + ".png")
        if not (os.path.exists(po) and os.path.exists(ps)):
            continue
        O = cv2.imread(po, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)
        S = cv2.imread(ps, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)
        m = O.max(2) >= 8
        L0, C0, h0 = rgb_to_lch(O)
        Lc, Cc, hc = rgb_to_lch(S)
        n = int(m.sum())
        if n < 1000:
            continue
        idx = rng.choice(n, min(n, 4000), replace=False)
        LOs.append(L0[m][idx]); LCs.append(Lc[m][idx]); C0s.append(C0[m][idx])
        used += 1
    if LOs:
        L0 = np.concatenate(LOs); Lc = np.concatenate(LCs); C0 = np.concatenate(C0s)
        A = np.c_[L0, np.ones(len(L0))]
        c, *_ = np.linalg.lstsq(A, Lc, rcond=None)
        pred = A @ c
        r2 = 1 - ((Lc - pred) ** 2).sum() / ((Lc - Lc.mean()) ** 2).sum()
        # chroma-stratified
        med = np.median(C0)
        lo = C0 <= med; hi = ~lo
        r2s = []
        for mask in [lo, hi]:
            A2 = np.c_[L0[mask], np.ones(mask.sum())]
            c2, *_ = np.linalg.lstsq(A2, Lc[mask], rcond=None)
            pr = A2 @ c2
            r2s.append(float(1 - ((Lc[mask] - pr) ** 2).sum() / ((Lc[mask] - Lc[mask].mean()) ** 2).sum()))
        res = Lc - pred
        out["h_L"] = {"n_images": used, "n_pixels": int(len(L0)),
                      "affine_slope": float(c[0]), "affine_intercept": float(c[1]), "r2": float(r2),
                      "r2_low_chroma": r2s[0], "r2_high_chroma": r2s[1],
                      "residual_std": float(res.std()), "residual_corr_chroma": float(np.corrcoef(res, C0)[0, 1])}
    json.dump(out, open(os.path.join(OUT, "e2_results.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

    # figure: affine offset contrast
    plt.figure(figsize=(4.5, 3))
    ks = np.array(out["affine_offset_contrast"], float)
    plt.plot(ks[:, 2], ks[:, 3], "o-")
    plt.plot([0, 0.5], [0, 0.5], "k--", lw=1)
    plt.xlabel("input normalized contrast K")
    plt.ylabel("output K' (a=0.8,b=10)")
    plt.title("Affine offset reduces normalized contrast")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "e2_affine_contrast.png"), dpi=180); plt.close()

    # figure: sinusoid response
    plt.figure(figsize=(5, 3))
    p = [x["period"] for x in sinus]
    for k, lab in [("affine_std", "T_hat"), ("gf_std", "GF r=64"), ("gauss_std", "Gauss sigma=21")]:
        plt.semilogx(p, [x[k] for x in sinus], "o-", label=lab)
    plt.xlabel("period (px)"); plt.ylabel("L* std")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, "e2_sinusoid.png"), dpi=180); plt.close()
    print("figures done")


if __name__ == "__main__":
    main()
