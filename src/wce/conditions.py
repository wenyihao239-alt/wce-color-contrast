"""Condition generators for the WCE 2x2 factorial pilot."""
import os, json, cv2, numpy as np
from .color import (apply_affine, rgb_to_lch, lch_to_srgb_gamut, rgb_to_lab, lab_to_rgb,
                    srgb_to_linear, linear_to_srgb)

RAW = r"C:\wce\data\raw"
RES = r"C:\wce\wce_experiment\results"
COND = r"C:\wce\wce_experiment\data\cond"
FULL_IMG = os.path.join(RAW, "seeai_full", "SEE_AI_project_all_images", "SEE_AI_project_all_images")
BG = cv2.imread(r"C:\wce\WCE2024\Retinanet\theBG.jpeg").astype(np.float64)
CCM_CCC = json.load(open(os.path.join(RES, "ccm_cccd.json")))
CCM_CC = json.load(open(os.path.join(RES, "ccm_ccd.json")))
M_CCC, B_CCC = np.array(CCM_CCC["M"]), np.array(CCM_CCC["b"])


def load_bgr(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(path)
    return im


def to_rgb(im):
    return im[:, :, ::-1].astype(np.float64)


def save_rgb(path, rgb, q=95):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.clip(np.round(rgb), 0, 255).astype(np.uint8)[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, q])


def save_bgr(path, bgr, q=95):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.clip(np.round(bgr), 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])


def valid_mask(A, thr=8.0):
    return A.max(axis=2) >= thr


def blend(A, out, mask):
    res = A.copy()
    res[mask] = out[mask]
    return res


def clahe_lab(rgb, clip=2.0, grid=8):
    lab = rgb_to_lab(rgb)
    L = np.clip(lab[..., 0], 0, 100)
    L8 = np.round(L * 2.55).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    Ln = cl.apply(L8).astype(np.float64) / 2.55
    lab2 = lab.copy(); lab2[..., 0] = Ln
    return lab_to_rgb(lab2, clip=True)


def guided_filter(I, r, eps=4.0):
    k = (2 * r + 1, 2 * r + 1)
    mean = cv2.boxFilter(I, -1, k, borderType=cv2.BORDER_REFLECT)
    var = cv2.boxFilter(I * I, -1, k, borderType=cv2.BORDER_REFLECT) - mean * mean
    a = var / (var + eps)
    b = mean - a * mean
    ma = cv2.boxFilter(a, -1, k, borderType=cv2.BORDER_REFLECT)
    mb = cv2.boxFilter(b, -1, k, borderType=cv2.BORDER_REFLECT)
    return ma * I + mb


def reliability(L, rgb, radius=2):
    # specular highlights: very bright, low chroma
    lab = rgb_to_lab(rgb)
    C = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    q_spec = 1.0 - 0.8 * ((L > 88) & (C < 18))
    # dark low-signal zones
    q_dark = 1.0 - 0.8 * (L < 12)
    # flat / low local SNR regions
    k = (2 * radius + 1, 2 * radius + 1)
    mu = cv2.boxFilter(L, -1, k, borderType=cv2.BORDER_REFLECT)
    sd = np.sqrt(np.maximum(cv2.boxFilter(L * L, -1, k, borderType=cv2.BORDER_REFLECT) - mu * mu, 0))
    q_flat = 1.0 - 0.6 * (sd < 2.0)
    q = q_spec * q_dark * q_flat
    return cv2.GaussianBlur(q, (0, 0), 2.0)


def make_conditions(A, C_official, r3=64, r_med=16, g_p2=1.0, g_p1=1.0, g_p1s=1.0, do_extra=False, e_clip=1.0, light=False):
    """A, C_official are RGB float arrays. Returns dict of generated RGB images."""
    m = valid_mask(A)
    Crec = blend(A, np.clip(apply_affine(A, M_CCC, B_CCC), 0, 255), m)
    Lc, Cc, hc = rgb_to_lch(Crec)
    L0, C0, h0 = rgb_to_lch(A)

    out = {}
    out["C"] = C_official if C_official is not None else Crec
    if not light:
        out["M1"] = blend(A, lch_to_srgb_gamut(Lc, C0, h0), m)
        out["M2"] = blend(A, lch_to_srgb_gamut(L0, Cc, hc), m)
        out["M3"] = blend(A, lch_to_srgb_gamut(Lc, Cc, hc), m)

    if not do_extra:
        return out

    Lbase = guided_filter(Lc, r3)
    L0_s = L0.copy()
    # detail bands of the original
    G1 = guided_filter(L0_s, max(1, r3 // 16))
    G2 = guided_filter(L0_s, max(1, r3 // 4))
    G3 = guided_filter(L0_s, r3)
    D_fine = L0_s - G1
    D_mid = G1 - G2
    D_coarse = G2 - G3
    D_all = D_fine + D_mid + D_coarse

    q = reliability(L0_s, A)
    Lp0 = Lbase + D_all
    out["P0"] = blend(A, lch_to_srgb_gamut(Lp0, Cc, hc), m)

    Lp2 = Lbase + g_p2 * q * D_all
    out["P2"] = blend(A, lch_to_srgb_gamut(Lp2, Cc, hc), m)

    out["E"] = clahe_lab(out["C"], e_clip, 8)
    if not light:
        Lp1 = Lbase + g_p1 * D_all
        out["P1"] = blend(A, lch_to_srgb_gamut(Lp1, Cc, hc), m)
        # single-scale ablation: detail only up to the median scale
        Gm = guided_filter(L0_s, max(1, r_med))
        D_single = L0_s - Gm
        Lp1s = Lbase + g_p1s * q * D_single
        out["P1s"] = blend(A, lch_to_srgb_gamut(Lp1s, Cc, hc), m)
        out["D"] = clahe_lab(A, e_clip, 8)
    return out
