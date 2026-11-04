import os, sys, csv, cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.color import rgb_to_lch, lch_to_srgb_gamut, rgb_to_lab, lab_to_rgb

ROOT = r"C:\wce\wce_experiment"
V2 = os.path.join(ROOT, "data", "v2")
LABELS = os.path.join(ROOT, "data", "labels")
MANIFEST = os.path.join(ROOT, "results", "manifest.csv")
LAM, ETA = 0.25, 1.0


def clahe(rgb):
    lab = rgb_to_lab(rgb)
    L = np.clip(lab[..., 0], 0, 100)
    L8 = np.round(L * 2.55).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    lab2 = lab.copy(); lab2[..., 0] = cl.apply(L8).astype(np.float64) / 2.55
    return lab_to_rgb(lab2, clip=True)


def save(path, rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.clip(np.round(rgb), 0, 255).astype(np.uint8)[:, :, ::-1])


def process(row):
    name, split = row["name"], row["split"]
    todo = [k for k in ["L", "X", "H", "E"]
            if not os.path.exists(os.path.join(V2, k, "images", split, name + ".png"))]
    if not todo:
        return name
    O = cv2.imread(os.path.join(V2, "O", "images", split, name + ".png"), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)
    S = cv2.imread(os.path.join(V2, "S", "images", split, name + ".png"), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)
    m = O.max(2) >= 8
    L0, C0, h0 = rgb_to_lch(O)
    Ls, Cs, hs = rgb_to_lch(S)
    if "L" in todo:
        im = O.copy(); im[m] = lch_to_srgb_gamut(Ls, C0, h0)[m]
        save(os.path.join(V2, "L", "images", split, name + ".png"), im)
    if "X" in todo:
        im = O.copy(); im[m] = lch_to_srgb_gamut(L0, Cs, hs)[m]
        save(os.path.join(V2, "X", "images", split, name + ".png"), im)
    if "H" in todo:
        L = (1 - LAM) * Ls + LAM * L0
        im = O.copy(); im[m] = lch_to_srgb_gamut(L, Cs, hs)[m]
        save(os.path.join(V2, "H", "images", split, name + ".png"), im)
    if "E" in todo:
        save(os.path.join(V2, "E", "images", split, name + ".png"), clahe(S))
    for k in todo:
        lp = os.path.join(V2, k, "labels", split, name + ".txt")
        src = os.path.join(LABELS, split, name + ".txt")
        if not os.path.exists(lp) and os.path.exists(src):
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            os.link(src, lp)
    return name


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    print("rows", len(rows), flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, _ in enumerate(ex.map(process, rows)):
            if i % 1000 == 0:
                print("done", i, flush=True)
    import yaml
    names = ['angiodysplasia', 'erosion', 'stenosis', 'lymphangiectasia', 'lymph-follicle', 'SMT',
             'polyp-like', 'bleeding', 'diverticulum', 'erythema', 'foreign-body', 'vein']
    for k in ["L", "X", "H", "E"]:
        root = os.path.join(V2, k)
        y = {"path": root, "train": "images/train", "val": "images/val", "test": "images/test",
             "nc": 12, "names": names}
        yaml.safe_dump(y, open(os.path.join(ROOT, "data", f"yaml_v2_{k}.yaml"), "w", encoding="utf-8"),
                       allow_unicode=True, sort_keys=False)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
