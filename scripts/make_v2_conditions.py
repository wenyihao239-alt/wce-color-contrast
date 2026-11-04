import os, sys, csv, json, argparse, cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.color import apply_affine, rgb_to_lch, lch_to_srgb_gamut, rgb_to_lab

RES = r"C:\wce\wce_experiment\results"
RAW = r"C:\wce\data\raw"
V2 = r"C:\wce\wce_experiment\data\v2"
FULL = os.path.join(RAW, "seeai_full", "SEE_AI_project_all_images", "SEE_AI_project_all_images")
BG = cv2.imread(r"C:\wce\WCE2024\Retinanet\theBG.jpeg").astype(np.float64)
CCM = json.load(open(os.path.join(RES, "v2_audit", "ccm_v2.json")))
M, B = np.array(CCM["M"]), np.array(CCM["b"])


def save_png(path, rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.clip(np.round(rgb), 0, 255).astype(np.uint8)[:, :, ::-1])


def load_o(name):
    im = cv2.imread(os.path.join(FULL, name + ".jpg"), cv2.IMREAD_COLOR).astype(np.float64)
    im = np.clip(np.round(im * BG / 255.0), 0, 255)
    return im[:, :, ::-1]


def process(row, extras):
    name, split = row["name"], row["split"]
    O = load_o(name)
    m = O.max(2) >= 8
    po = os.path.join(V2, "O", split, name + ".png")
    if not os.path.exists(po):
        save_png(po, O)
    L0, C0, h0 = rgb_to_lch(O)
    un = apply_affine(O, M, B)
    Lc, Cc, hc = rgb_to_lch(un)
    ps = os.path.join(V2, "S", split, name + ".png")
    if not os.path.exists(ps):
        S = O.copy(); S[m] = lch_to_srgb_gamut(Lc, Cc, hc)[m]
        save_png(ps, S)
    if extras and split == "val":
        pl = os.path.join(V2, "L", split, name + ".png")
        if not os.path.exists(pl):
            im = O.copy(); im[m] = lch_to_srgb_gamut(Lc, C0, h0)[m]; save_png(pl, im)
        px = os.path.join(V2, "X", split, name + ".png")
        if not os.path.exists(px):
            im = O.copy(); im[m] = lch_to_srgb_gamut(L0, Cc, hc)[m]; save_png(px, im)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(RES, "manifest.csv"), encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]
    print("rows", len(rows), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, _ in enumerate(ex.map(lambda r: process(r, True), rows)):
            if i % 1000 == 0:
                print("done", i, flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
