import zipfile, cv2, numpy as np, os
from scipy.optimize import least_squares

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
names = sorted(set(eA) & set(e1) & set(e2))[:12]
rng = np.random.default_rng(0)


def srgb2lin(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255.0


def read(z, e):
    return cv2.imdecode(np.frombuffer(z.read(e), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)


def jpeg_noise(im, q=95):
    ok, buf = cv2.imencode('.jpg', im[:, :, ::-1].astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)
    return np.sqrt(((dec - im) ** 2).mean())


print("--- JPEG noise / clipping ---")
for tag, z, es in [('A', zA, eA), ('C1', z1, e1), ('C2', z2, e2)]:
    ns = []; clip = []
    for n in names:
        im = read(z, es[n])
        ns.append(jpeg_noise(im))
        clip.append(((im.max(2) >= 254).mean() * 100))
    print(f"{tag}: jpeg95_rmse={np.mean(ns):.2f}  pct_pixels_max>=254={np.mean(clip):.2f}%")

print("--- affine fits (gamma space) ---")
for tag, z, es in [('C1', z1, e1), ('C2', z2, e2)]:
    coefs = []
    for n in names:
        A = read(zA, eA[n]); C = read(z, es[n])
        good = (C.min(2) > 2) & (C.max(2) < 253) & (A.min(2) > 2) & (A.max(2) < 253)
        X = A[good]; Y = C[good]
        if len(X) > 50000:
            idx = rng.choice(len(X), 50000, replace=False)
            X = X[idx]; Y = Y[idx]
        X1 = np.c_[X, np.ones(len(X))]
        coef, *_ = np.linalg.lstsq(X1, Y, rcond=None)
        coefs.append(coef)
    cm = np.mean(coefs, axis=0)
    print(tag, "M=\n", cm[:3].round(5), "\noffset=", cm[3].round(3))
    print(tag, "row sums (with offset/255):", (cm[:3].sum(0) + cm[3] / 255).round(3))
