import os, sys, csv, json, argparse, cv2, numpy as np
from multiprocessing import Pool
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.conditions import M_CCC, B_CCC
from wce.color import apply_affine, rgb_to_lab, srgb_to_linear

RES = r"C:\wce\wce_experiment\results"
COND = r"C:\wce\wce_experiment\data\cond"
MANIFEST = os.path.join(RES, "manifest.csv")
LUMA = np.array([0.2126, 0.7152, 0.0722])


def read_rgb(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)


def labels_for(row):
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


def g1_worker(row):
    pa = os.path.join(COND, "A", row["split"], row["name"] + ".jpg")
    pc = os.path.join(COND, "C", row["split"], row["name"] + ".jpg")
    if not (os.path.exists(pa) and os.path.exists(pc)):
        return None
    A = read_rgb(pa)
    C = read_rgb(pc)
    m = A.max(2) >= 8
    if m.sum() < 100:
        return None
    un = apply_affine(A, M_CCC, B_CCC)
    anyclip = ((un > 255.0) & m[..., None]).any(2)
    rclip = (un[..., 0] > 255.0) & m
    actual_any = ((C >= 254) & m[..., None]).any(2)
    actual_r = (C[..., 0] >= 254) & m
    return {"valid": int(m.sum()),
            "pot_any": int(anyclip.sum()), "pot_r": int(rclip.sum()),
            "act_any": int(actual_any.sum()), "act_r": int(actual_r.sum())}


def g2_worker(row):
    pa = os.path.join(COND, "A", row["split"], row["name"] + ".jpg")
    pc = os.path.join(COND, "C", row["split"], row["name"] + ".jpg")
    if not (os.path.exists(pa) and os.path.exists(pc)):
        return None
    A = read_rgb(pa)
    C = read_rgb(pc)
    boxes = labels_for(row)
    if not boxes:
        return None
    LA = rgb_to_lab(A)[..., 0]; LC = rgb_to_lab(C)[..., 0]
    gA = srgb_to_linear(A)[..., 1]; gC = srgb_to_linear(C)[..., 1]
    # luminance in linear space
    yA = (srgb_to_linear(A) @ LUMA); yC = (srgb_to_linear(C) @ LUMA)
    def grads(x):
        gx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3); gy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)
        return np.hypot(gx, gy)
    out = []
    for (c, cx, cy, bw, bh) in boxes:
        x1 = int(max(0, (cx - bw / 2) * 576)); x2 = int(min(575, (cx + bw / 2) * 576))
        y1 = int(max(0, (cy - bh / 2) * 576)); y2 = int(min(575, (cy + bh / 2) * 576))
        pad = 6
        ex1, ex2 = max(0, x1 - pad), min(576, x2 + pad)
        ey1, ey2 = max(0, y1 - pad), min(576, y2 + pad)
        if ex2 - ex1 < 5 or ey2 - ey1 < 5:
            continue
        reg = (slice(ey1, ey2), slice(ex1, ex2))
        # ring only
        mask = np.ones((ey2 - ey1, ex2 - ex1), bool)
        mask[y1 - ey1:y2 - ey1, x1 - ex1:x2 - ex1] = False
        if mask.sum() < 10:
            continue
        out.append({
            "gA": float(grads(gA)[reg][mask].mean()), "gC": float(grads(gC)[reg][mask].mean()),
            "yA": float(grads(yA)[reg][mask].mean()), "yC": float(grads(yC)[reg][mask].mean()),
            "LA": float(grads(LA)[reg][mask].mean()), "LC": float(grads(LC)[reg][mask].mean()),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="g1")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]
    if args.what == "g1":
        with Pool(args.workers) as p:
            res = [x for x in p.imap_unordered(g1_worker, rows, chunksize=16) if x]
        agg = {k: sum(x[k] for x in res) for k in res[0]}
        valid = agg["valid"]
        out = {"n_images": len(res), "valid_pixels": valid,
               "potential_clip_any_pct": 100 * agg["pot_any"] / valid,
               "potential_clip_R_pct": 100 * agg["pot_r"] / valid,
               "actual_clip_any_pct": 100 * agg["act_any"] / valid,
               "actual_clip_R_pct": 100 * agg["act_r"] / valid}
        json.dump(out, open(os.path.join(RES, "gate_g1.json"), "w"), indent=2)
        print(json.dumps(out, indent=2))
    elif args.what == "g2":
        w = {}
        w["accum_ccc"] = (LUMA @ np.array(M_CCC)).tolist()
        w["accum_cc"] = (LUMA @ np.array(json.load(open(os.path.join(RES, "ccm_ccd.json")))["M"])).tolist()
        for k in ["accum_ccc", "accum_cc"]:
            v = np.array(w[k]); w[k + "_normalised"] = (v / v.sum()).tolist()
        with Pool(args.workers) as p:
            res = [x for x in p.imap_unordered(g2_worker, rows, chunksize=8) if x]
        flat = [y for xs in res for y in xs]
        if flat:
            w["n_boxes"] = len(flat)
            w["grad_ratio_A_GB"] = float(np.median([x["gA"] / (x["yA"] + 1e-9) for x in flat]))
            w["grad_ratio_C_GB"] = float(np.median([x["gC"] / (x["yC"] + 1e-9) for x in flat]))
            w["grad_ratio_A_L"] = float(np.median([x["LA"] for x in flat]))
            w["grad_ratio_C_L"] = float(np.median([x["LC"] for x in flat]))
            w["green_grad_A"] = float(np.median([x["gA"] for x in flat]))
            w["green_grad_C"] = float(np.median([x["gC"] for x in flat]))
            w["lum_grad_A"] = float(np.median([x["yA"] for x in flat]))
            w["lum_grad_C"] = float(np.median([x["yC"] for x in flat]))
        json.dump(w, open(os.path.join(RES, "gate_g2.json"), "w"), indent=2)
        print(json.dumps(w, indent=2))


if __name__ == "__main__":
    main()
