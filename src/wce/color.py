"""Colour-space utilities for the WCE pilot study.

All RGB arrays are float64 in 0..255 sRGB (gamma-encoded) unless a function
name contains 'linear'.  Lab/LCh use D65 / 2 deg sRGB.
"""
import numpy as np

SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ)
WHITE = np.array([0.95047, 1.00000, 1.08883])
EPS = 216.0 / 24389.0
KAPPA = 24389.0 / 27.0


def srgb_to_linear(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255.0


def _f(t):
    return np.where(t > EPS, np.cbrt(t), (KAPPA * t + 16.0) / 116.0)


def _finv(t):
    return np.where(t ** 3 > EPS, t ** 3, (116.0 * t - 16.0) / KAPPA)


def rgb_to_lab(rgb):
    lin = srgb_to_linear(rgb)
    xyz = lin @ SRGB_TO_XYZ.T
    fx, fy, fz = _f(xyz[..., 0] / WHITE[0]), _f(xyz[..., 1] / WHITE[1]), _f(xyz[..., 2] / WHITE[2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab, clip=True):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xyz = np.stack([WHITE[0] * _finv(fx), WHITE[1] * _finv(fy), WHITE[2] * _finv(fz)], axis=-1)
    lin = xyz @ XYZ_TO_SRGB.T
    if clip:
        lin = np.clip(lin, 0.0, 1.0)
    return linear_to_srgb(lin)


def rgb_to_lch(rgb):
    lab = rgb_to_lab(rgb)
    L = lab[..., 0]
    C = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0
    return L, C, h


def lch_to_lab(L, C, h):
    rad = np.radians(h)
    return np.stack([L, C * np.cos(rad), C * np.sin(rad)], axis=-1)


def in_gamut(rgb, tol=1e-6):
    return (rgb[..., 0] >= -tol) & (rgb[..., 0] <= 255 + tol) & \
           (rgb[..., 1] >= -tol) & (rgb[..., 1] <= 255 + tol) & \
           (rgb[..., 2] >= -tol) & (rgb[..., 2] <= 255 + tol)


def lch_to_srgb_gamut(L, C, h, iters=14):
    """Convert LCh to sRGB, reducing chroma only (constant L and h) when out of gamut."""
    L = np.asarray(L, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    rgb = lab_to_rgb(lch_to_lab(L, C, h), clip=False)
    oog = ~in_gamut(rgb)
    if oog.any():
        lo = np.zeros_like(C)
        hi = C.copy()
        Lt, ht = L[oog], h[oog]
        for _ in range(iters):
            mid = 0.5 * (lo[oog] + hi[oog])
            trial = lab_to_rgb(lch_to_lab(Lt, mid, ht), clip=False)
            ok = in_gamut(trial)
            lo[oog] = np.where(ok, mid, lo[oog])
            hi[oog] = np.where(ok, hi[oog], mid)
        rgb[oog] = lab_to_rgb(lch_to_lab(Lt, lo[oog], ht), clip=True)
    return np.clip(rgb, 0.0, 255.0)


def delta_e_2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    """CIEDE2000, vectorised; lab arrays (...,3)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    Cb = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where((C1p * C2p) == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))
    Lbp = (L1 + L2) / 2.0
    Cbp = (C1p + C2p) / 2.0
    hbp = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) > 180, hbp + 360, hbp)
    hbp = np.where((C1p * C2p) == 0, h1p + h2p, hbp) / 2.0
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-((hbp - 275) / 25.0) ** 2)
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / (kL * Sl)) ** 2 + (dCp / (kC * Sc)) ** 2 + (dHp / (kH * Sh)) ** 2
                   + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh)))


def apply_affine(rgb, M, b):
    """Apply a gamma-space affine colour transform without clipping."""
    return rgb @ np.asarray(M).T + np.asarray(b)
