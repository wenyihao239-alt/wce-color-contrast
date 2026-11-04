import os, json, cv2, numpy as np

RAW = r"C:\wce\data\raw"
OUT = r"C:\wce\wce_experiment\results"
os.makedirs(OUT, exist_ok=True)


def load(split, name):
    p = os.path.join(RAW, split, "images", "training", name)
    return cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)


def blockmean(im, k=4):
    h, w = im.shape[:2]
    return im[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k, 3).mean((1, 3))


def fit_ccm(tag):
    """tag: ccd / cccd"""
    names = sorted(os.listdir(os.path.join(RAW, "origd", "images", "training")))
    rng = np.random.default_rng(42)
    Xs, Ys = [], []
    picked = names[::13]
    for n in picked:
        A = blockmean(load("origd", n))
        C = blockmean(load(tag, n))
        good = (C.min(2) > 2) & (C.max(2) < 253) & (A.min(2) > 2) & (A.max(2) < 253)
        good = cv2.erode(good.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        X = A[good]; Y = C[good]
        if len(X) < 100:
            continue
        idx = rng.choice(len(X), min(len(X), 1500), replace=False)
        Xs.append(X[idx]); Ys.append(Y[idx])
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    X1 = np.c_[X, np.ones(len(X))]
    coef, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    M = coef[:3].T  # conventional 3x3, output x input
    b = coef[3]
    pred = X1 @ coef
    rmse = float(np.sqrt(((pred - Y) ** 2).mean()))
    d = {"M": M.tolist(), "b": b.tolist(), "rmse_block": rmse, "n": int(len(X)), "n_images": len(picked)}
    with open(os.path.join(OUT, f"ccm_{tag}.json"), "w") as f:
        json.dump(d, f, indent=2)
    print(tag, "RMSE", round(rmse, 3), "\nM=\n", M.round(5), "\nb=", b.round(3), "rowsum=", (M.sum(0) + b / 255).round(3))
    return d


if __name__ == "__main__":
    fit_ccm("cccd")
    fit_ccm("ccd")
