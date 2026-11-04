import zipfile, cv2, numpy as np, os, json

BASE = r"C:\Users\YOLO\Documents\胶囊内镜\data\downloads"
z = {"origd": zipfile.ZipFile(os.path.join(BASE, "archive.zip")),
     "ccd": zipfile.ZipFile(os.path.join(BASE, "archive (1).zip")),
     "cccd": zipfile.ZipFile(os.path.join(BASE, "archive (2).zip"))}
roots = {"origd": "R_OrigD/", "ccd": "data/", "cccd": "data/"}


def entries(zf, root):
    return {os.path.basename(e.filename): e for e in zf.infolist()
            if e.filename.lower().endswith('.jpg') and f"{root}images/training/" in e.filename}


E = {k: entries(v, roots[k]) for k, v in z.items()}
names = sorted(set(E['origd']) & set(E['ccd']) & set(E['cccd']))
rng = np.random.default_rng(42)
print("train pairs", len(names))


def read(k, n):
    return cv2.imdecode(np.frombuffer(z[k].read(E[k][n]), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)


def blockmean(im, k=4):
    h, w = im.shape[:2]
    return im[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k, 3).mean((1, 3))


pairs = []
for n in names[::7][:400]:
    A = blockmean(read('origd', n)); C = blockmean(read('cccd', n))
    good = (C.min(2) > 2) & (C.max(2) < 253) & (A.min(2) > 2) & (A.max(2) < 253)
    good = cv2.erode(good.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    X = A[good]; Y = C[good]
    if len(X):
        idx = rng.choice(len(X), min(len(X), 4000), replace=False)
        pairs.append((X[idx], Y[idx]))
X = np.concatenate([p[0] for p in pairs]); Y = np.concatenate([p[1] for p in pairs])
print("samples", len(X))
X1 = np.c_[X, np.ones(len(X))]
c, *_ = np.linalg.lstsq(X1, Y, rcond=None)
pred = X1 @ c
print("affine gamma RMSE", np.sqrt(((pred - Y) ** 2).mean()))
print("M=\n", c[:3]); print("b=", c[3])
# per-channel quadratic
for ch in range(3):
    feats = [np.ones(len(X)), X[:, 0], X[:, 1], X[:, 2],
             X[:, 0] ** 2, X[:, 1] ** 2, X[:, 2] ** 2, X[:, 0] * X[:, 1], X[:, 0] * X[:, 2], X[:, 1] * X[:, 2]]
    F = np.stack(feats, 1)
    cc, *_ = np.linalg.lstsq(F, Y[:, ch], rcond=None)
    print("quad ch", ch, "RMSE", np.sqrt(((F @ cc - Y[:, ch]) ** 2).mean()))
# no-offset
M, *_ = np.linalg.lstsq(X, Y, rcond=None)
print("no-offset gamma RMSE", np.sqrt(((X @ M - Y) ** 2).mean()))
