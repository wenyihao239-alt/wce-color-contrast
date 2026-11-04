import zipfile, cv2, numpy as np, os
from scipy.optimize import least_squares

BASE = r"C:\Users\YOLO\Documents\胶囊内镜\data\downloads"
zA = zipfile.ZipFile(os.path.join(BASE, "archive.zip"))
z2 = zipfile.ZipFile(os.path.join(BASE, "archive (2).zip"))  # CCCD


def entries(z, root):
    return {os.path.basename(e.filename): e for e in z.infolist()
            if e.filename.lower().endswith('.jpg') and e.filename.startswith(root)}


eA = entries(zA, 'R_OrigD/images/training/')
eC = entries(z2, 'data/images/training/')
names = sorted(set(eA) & set(eC))[:16]
rng = np.random.default_rng(0)


def srgb2lin(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255.0


def read(z, e):
    return cv2.imdecode(np.frombuffer(z.read(e), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)


Xs, Ys = [], []
for n in names:
    A = read(zA, eA[n]); C = read(z2, eC[n])
    good = (C.min(2) > 2) & (C.max(2) < 253) & (A.min(2) > 2) & (A.max(2) < 253)
    X = A[good]; Y = C[good]
    if len(X) > 30000:
        idx = rng.choice(len(X), 30000, replace=False)
        X = X[idx]; Y = Y[idx]
    Xs.append(X); Ys.append(Y)
X = np.concatenate(Xs); Y = np.concatenate(Ys)
print("samples", len(X))


def rmse(p):
    return np.sqrt(np.mean((p - Y) ** 2))


def fit_gamma_affine():
    X1 = np.c_[X, np.ones(len(X))]
    c, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    return rmse(X1 @ c), c


def fit_model(space, affine):
    Xl = srgb2lin(X) if space == 'linear' else X
    n = 9 + (3 if affine else 0)

    def resid(p):
        M = p[:9].reshape(3, 3)
        b = p[9:12] if affine else np.zeros(3)
        pred = Xl @ M.T + b
        if space == 'linear':
            pred = lin2srgb(pred)
        else:
            pred = np.clip(pred, 0, 255)
        return (pred - Y).ravel()

    p0 = np.r_[np.eye(3).ravel(), np.zeros(3)]
    r = least_squares(resid, p0, loss='soft_l1', f_scale=3.0, max_nfev=200, verbose=0)
    return np.sqrt(np.mean(r.fun ** 2)), r.x[:9].reshape(3, 3), (r.x[9:12] if affine else np.zeros(3))


r, c = fit_gamma_affine()
print(f"gamma affine global: RMSE={r:.3f}\nM=\n{c[:3].round(5)}\nb={c[3].round(3)}")
for space in ['gamma', 'linear']:
    for affine in [False, True]:
        rr, M, b = fit_model(space, affine)
        print(f"nonlinear {space} affine={affine}: RMSE={rr:.3f}\nM=\n{M.round(4)}\nb={b.round(3)}")
