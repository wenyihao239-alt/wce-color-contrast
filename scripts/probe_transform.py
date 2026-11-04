import zipfile, cv2, numpy as np, os, sys

BASE = r"C:\Users\YOLO\Documents\胶囊内镜\data\downloads"
zA = zipfile.ZipFile(os.path.join(BASE, "archive.zip"))
z1 = zipfile.ZipFile(os.path.join(BASE, "archive (1).zip"))
z2 = zipfile.ZipFile(os.path.join(BASE, "archive (2).zip"))


def entries(z, root):
    return {os.path.basename(e.filename): e for e in z.infolist()
            if e.filename.lower().endswith('.jpg') and e.filename.startswith(root)}


eA = entries(zA, 'R_OrigD/images/training/')
e1 = entries(z1, 'data/images/training/')
e2 = entries(z2, 'data/images/training/')
names = sorted(set(eA) & set(e1) & set(e2))[:10]
print("pairs", len(names))


def srgb2lin(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255.0


def read(z, e):
    return cv2.imdecode(np.frombuffer(z.read(e), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)


rng = np.random.default_rng(0)
for tag, z, es in [('C1(287MB)', z1, e1), ('C2(267MB)', z2, e2)]:
    for space in ['gamma', 'linear']:
        for affine in [False, True]:
            errs = []; gains = []
            coefs = []
            for n in names:
                A = read(zA, eA[n]); C = read(z, es[n])
                good = (C.min(2) > 2) & (C.max(2) < 253) & (A.min(2) > 2) & (A.max(2) < 253)
                X = A[good]; Yraw = C[good]
                if len(X) > 80000:
                    idx = rng.choice(len(X), 80000, replace=False)
                    X = X[idx]; Yraw = Yraw[idx]
                if space == 'linear':
                    X = srgb2lin(X); Y = srgb2lin(Yraw)
                else:
                    Y = Yraw
                X1 = np.c_[X, np.ones(len(X))] if affine else X
                coef, *_ = np.linalg.lstsq(X1, Y, rcond=None)
                pred = X1 @ coef
                if space == 'linear':
                    pred = lin2srgb(pred)
                errs.append(np.sqrt(((pred - Yraw) ** 2).mean()))
                coefs.append(coef[:3])
            print(f"{tag} {space:6s} affine={affine} RMSE={np.mean(errs):.2f}")
            if space == 'gamma' and affine:
                print("  mean matrix:\n", np.mean(coefs, axis=0).round(4))
