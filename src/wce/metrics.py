"""Image- and box-level metrics used by the zero-cost gates and sweep."""
import numpy as np, cv2
from .color import rgb_to_lab, rgb_to_lch, srgb_to_linear, delta_e_2000


def masks(rgb):
    A = rgb
    valid = A.max(axis=2) >= 8
    lab = rgb_to_lab(rgb)
    L = lab[..., 0]
    C = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    spec = (L > 88) & (C < 18)
    dark = L < 10
    mucosa = valid & ~spec & ~dark
    return valid, mucosa, L, lab


def local_rms(L, mucosa, k=9):
    mu = cv2.boxFilter(L, -1, (k, k), borderType=cv2.BORDER_REFLECT)
    sd = np.sqrt(np.maximum(cv2.boxFilter(L * L, -1, (k, k), borderType=cv2.BORDER_REFLECT) - mu * mu, 0))
    return float(sd[mucosa].mean()) if mucosa.any() else np.nan


def box_ls(L, valid, boxes, pad_frac=0.25, pad_min=6):
    """Lesion-surround contrasts for YOLO boxes (cx,cy,w,h normalised)."""
    H, W = L.shape
    vals = []
    for (cx, cy, bw, bh) in boxes:
        x1 = int(round((cx - bw / 2) * W)); x2 = int(round((cx + bw / 2) * W))
        y1 = int(round((cy - bh / 2) * H)); y2 = int(round((cy + bh / 2) * H))
        x1, x2 = max(0, x1), min(W - 1, x2)
        y1, y2 = max(0, y1), min(H - 1, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        inner = L[y1:y2, x1:x2][valid[y1:y2, x1:x2]]
        px = max(pad_min, int(round(pad_frac * (x2 - x1))))
        py = max(pad_min, int(round(pad_frac * (y2 - y1))))
        ex1, ex2 = max(0, x1 - px), min(W, x2 + px)
        ey1, ey2 = max(0, y1 - py), min(H, y2 + py)
        ring = L[ey1:ey2, ex1:ex2].copy()
        vm = valid[ey1:ey2, ex1:ex2]
        ring_mask = vm.copy()
        ring_mask[y1 - ey1:y2 - ey1, x1 - ex1:x2 - ex1] = False
        ring = ring[ring_mask]
        if inner.size < 8 or ring.size < 8:
            continue
        mi, mr = inner.mean(), ring.mean()
        michelson = abs(mi - mr) / (mi + mr + 1e-6)
        weber = abs(mi - mr) / (mr + 1e-6)
        vals.append((michelson, weber, mi, mr))
    return np.array(vals) if vals else np.zeros((0, 4))


def de_stats(rgb, ref_lab, mucosa, signed=False):
    lab = rgb_to_lab(rgb)
    de = delta_e_2000(lab, ref_lab)
    v = de[mucosa]
    return {"dE_median": float(np.median(v)), "dE_mean": float(v.mean()),
            "dE_p95": float(np.percentile(v, 95))}
