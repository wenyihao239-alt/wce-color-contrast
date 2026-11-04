import os, sys, csv, json, cv2, numpy as np
from multiprocessing import Pool
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.color import rgb_to_lab
from wce.metrics import box_ls

RES = r"C:\wce\wce_experiment\results"
COND = r"C:\wce\wce_experiment\data\cond"
MANIFEST = os.path.join(RES, "manifest.csv")
CONDS = ["A", "M1", "M2", "M3", "C"]


def labels(row):
    n = row["name"]
    if row["source"] == "origd":
        sp = "training" if row["official_split"] == "train" else "validation"
        p = os.path.join(r"C:\wce\data\raw\origd", "labels", sp, n + ".txt")
    else:
        p = os.path.join(r"C:\wce\data\raw\seeai_full", "SEE_AI_project_all_txt", "SEE_AI_project_all_txt", n + ".txt")
    out = []
    if os.path.exists(p):
        for line in open(p):
            q = line.split()
            if len(q) == 5:
                out.append((int(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4])))
    return out


def worker(row):
    boxes = labels(row)
    if not boxes:
        return None
    acc = {c: [0.0, 0.0, 0] for c in CONDS}
    cls_acc = {c: {} for c in CONDS}
    A = cv2.imread(os.path.join(COND, "A", row["split"], row["name"] + ".jpg"), cv2.IMREAD_COLOR)[:, :, ::-1]
    m = A.max(2) >= 8
    b = [(cx, cy, w, h) for (_, cx, cy, w, h) in boxes]
    classes = [x[0] for x in boxes]
    for c in CONDS:
        im = A if c == "A" else cv2.imread(os.path.join(COND, c, row["split"], row["name"] + ".jpg"), cv2.IMREAD_COLOR)
        if im is None:
            continue
        if c != "A":
            im = im[:, :, ::-1]
        L = rgb_to_lab(im.astype(np.float64))[..., 0]
        vals = box_ls(L, m, b)
        if len(vals) == 0:
            continue
        for j, (mic, web, mi, mr) in enumerate(vals):
            k = classes[j]
            acc[c][0] += mic; acc[c][1] += web; acc[c][2] += 1
            cls_acc[c].setdefault(k, [0.0, 0.0, 0])
            cls_acc[c][k][0] += mic; cls_acc[c][k][1] += web; cls_acc[c][k][2] += 1
    return acc, cls_acc


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    total = {c: [0.0, 0.0, 0] for c in CONDS}
    cls_total = {c: {} for c in CONDS}
    with Pool(6) as p:
        for i, res in enumerate(p.imap_unordered(worker, rows, chunksize=16)):
            if res is None:
                continue
            acc, ca = res
            for c in CONDS:
                total[c][0] += acc[c][0]; total[c][1] += acc[c][1]; total[c][2] += acc[c][2]
                for k, v in ca[c].items():
                    cls_total[c].setdefault(k, [0.0, 0.0, 0])
                    cls_total[c][k][0] += v[0]; cls_total[c][k][1] += v[1]; cls_total[c][k][2] += v[2]
            if i % 2000 == 0:
                print("done", i, flush=True)
    names = ['angiodysplasia', 'erosion', 'stenosis', 'lymphangiectasia', 'lymph-follicle', 'SMT',
             'polyp-like', 'bleeding', 'diverticulum', 'erythema', 'foreign-body', 'vein']
    out = {"overall": {}, "classes": {}}
    for c in CONDS:
        s = total[c]
        out["overall"][c] = {"michelson": s[0] / max(s[2], 1), "weber": s[1] / max(s[2], 1), "n": int(s[2])}
        out["classes"][c] = {}
        for k, v in cls_total[c].items():
            out["classes"][c][names[k]] = {"michelson": v[0] / max(v[2], 1), "weber": v[1] / max(v[2], 1), "n": int(v[2])}
    json.dump(out, open(os.path.join(RES, "gate_g3.json"), "w"), indent=2)
    print(json.dumps(out["overall"], indent=2))


if __name__ == "__main__":
    main()
