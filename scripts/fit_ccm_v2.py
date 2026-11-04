import os, sys, csv, json, hashlib, cv2, numpy as np
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.color import apply_affine, rgb_to_lab, delta_e_2000, srgb_to_linear

RES = r"C:\wce\wce_experiment\results"
RAW = r"C:\wce\data\raw"
OUT = os.path.join(RES, "v2_audit")
os.makedirs(OUT, exist_ok=True)
FULL = os.path.join(RAW, "seeai_full", "SEE_AI_project_all_images", "SEE_AI_project_all_images")
BG = cv2.imread(r"C:\wce\WCE2024\Retinanet\theBG.jpeg").astype(np.float64)


def load_o(row):
    im = cv2.imread(os.path.join(FULL, row["name"] + ".jpg"), cv2.IMREAD_COLOR).astype(np.float64)
    im = np.clip(np.round(im * BG / 255.0), 0, 255)
    return im[:, :, ::-1]  # RGB


def load_cpub(row):
    sp = "training" if row["official_split"] == "train" else "validation"
    p = os.path.join(RAW, "cccd", "images", sp, row["name"] + ".jpg")
    return cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)


def blockmean(im, k=4):
    h, w = im.shape[:2]
    return im[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k, 3).mean((1, 3))


def main():
    rows = list(csv.DictReader(open(os.path.join(RES, "manifest.csv"), encoding="utf-8")))
    man = open(os.path.join(RES, "manifest.csv"), "rb").read()
    frozen = {"sha256": hashlib.sha256(man).hexdigest(), "n": len(rows)}
    json.dump(frozen, open(os.path.join(OUT, "manifest_frozen.json"), "w"), indent=2)
    train = [r for r in rows if r["split"] == "train" and r["source"] == "origd"]
    val = [r for r in rows if r["split"] == "val" and r["source"] == "origd"]
    rng = np.random.default_rng(42)
    Xs, Ys = [], []
    picked = train[::9]
    for r in picked:
        O = blockmean(load_o(r)); C = blockmean(load_cpub(r))
        good = (C.min(2) > 2) & (C.max(2) < 253) & (O.min(2) > 2) & (O.max(2) < 253)
        good = cv2.erode(good.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        X, Y = O[good], C[good]
        if len(X) < 100:
            continue
        idx = rng.choice(len(X), min(len(X), 1500), replace=False)
        Xs.append(X[idx]); Ys.append(Y[idx])
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    coef, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X))], Y, rcond=None)
    M = coef[:3].T; b = coef[3]
    pred = X @ M.T + b
    rmse = float(np.sqrt(((pred - Y) ** 2).mean()))
    # validation ΔE on a few images
    des = []
    neg = 0; pos = 0; n = 0
    for r in val[::20]:
        O = load_o(r); C = load_cpub(r)
        m = O.max(2) >= 8
        un = apply_affine(O, M, b)
        neg += int((un < 0).any(2)[m].sum()); pos += int((un > 255).any(2)[m].sum()); n += int(m.sum())
        rec = O.copy(); rec[m] = np.clip(un[m], 0, 255)
        de = delta_e_2000(rgb_to_lab(rec), rgb_to_lab(C))
        des.append(np.median(de[m]))
    des = np.array(des)
    # linear-space luminance weights with derivative of decode at mid-grey
    def dec(x):
        x = x / 255.0
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    def ddec(x):
        x = x / 255.0
        return np.where(x <= 0.04045, 1 / 12.92, 2.4 * ((x + 0.055) / 1.055) ** 1.4 / 1.055) / 255.0
    luma = np.array([0.2126, 0.7152, 0.0722])
    x0 = np.array([128.0, 128.0, 128.0])
    J = luma @ np.diag(ddec(x0)) @ M
    out = {"M": M.tolist(), "b": b.tolist(), "rmse_block": rmse,
           "val_dE_median": float(np.median(des)), "val_dE_p95": float(np.percentile(des, 95)),
           "n_train_images": len(picked), "n_fit_pixels": int(len(X)),
           "negative_channel_rate_pct": 100 * neg / max(n, 1), "over255_rate_pct": 100 * pos / max(n, 1),
           "luma_derivative_weights_midgrey": J.tolist(),
           "row_sum_with_offset": (M.sum(1) + b / 255).tolist()}
    json.dump(out, open(os.path.join(OUT, "ccm_v2.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
