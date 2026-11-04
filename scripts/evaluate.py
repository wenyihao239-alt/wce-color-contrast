import os, sys, csv, json, argparse, cv2, numpy as np, torch
from collections import defaultdict
sys.path.insert(0, r"C:\wce\WCE2024\Yolov5\yolov5")
from models.experimental import attempt_load
from utils.general import non_max_suppression
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.data import CLASS_NAMES

RES = r"C:\wce\wce_experiment\results"
COND = r"C:\wce\wce_experiment\data\cond"
RUNS = r"C:\wce\wce_experiment\runs\controlled"
MANIFEST = os.path.join(RES, "manifest.csv")


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def load_gts(name):
    row = ROW[name]
    if row["source"] == "origd":
        sp = "training" if row["official_split"] == "train" else "validation"
        p = os.path.join(r"C:\wce\data\raw\origd", "labels", sp, name + ".txt")
    else:
        p = os.path.join(r"C:\wce\data\raw\seeai_full", "SEE_AI_project_all_txt", "SEE_AI_project_all_txt", name + ".txt")
    out = []
    if os.path.exists(p):
        for line in open(p):
            q = line.split()
            if len(q) == 5:
                c, cx, cy, w, h = int(q[0]), *map(float, q[1:])
                out.append([c, (cx - w / 2) * 576, (cy - h / 2) * 576, (cx + w / 2) * 576, (cy + h / 2) * 576])
    return np.array(out, float) if out else np.zeros((0, 5))


def match_image(dets, gts, iou_thr):
    """dets: Nx6 xyxy conf cls; gts: Mx5 cls xyxy. Returns TP flags and matched gt idx."""
    tp = np.zeros(len(dets), bool); mg = -np.ones(len(dets), int)
    for c in range(12):
        di = np.where(dets[:, 5] == c)[0]
        gi = np.where(gts[:, 0] == c)[0]
        if len(di) == 0:
            continue
        if len(gi) == 0:
            continue
        order = di[np.argsort(-dets[di, 4])]
        ious = iou_matrix(dets[order, :4], gts[gi, 1:5])
        used = np.zeros(len(gi), bool)
        for k, idx in enumerate(order):
            if used.all():
                break
            j = int(np.argmax(np.where(used, -1, ious[k])))
            if ious[k, j] >= iou_thr:
                used[j] = True; tp[idx] = True; mg[idx] = gi[j]
    return tp, mg


def ap_from_dets(confs, tps, npos, npoints=101):
    if npos == 0:
        return np.nan
    if len(confs) == 0:
        return 0.0
    order = np.argsort(-confs)
    tps = tps[order]; confs = confs[order]
    ctp = np.cumsum(tps); cfp = np.cumsum(~tps)
    rec = ctp / npos; prec = ctp / (ctp + cfp + 1e-9)
    mrec = np.linspace(0, 1, npoints)
    interp = np.interp(mrec, rec, prec)
    return float(interp.mean())


def evaluate_model(weights, cond, split="test", conf=0.24, batch=16, device="cuda:0"):
    model = attempt_load(weights, device=device)
    model.eval().float()
    rows = [r for r in CSV if r["split"] == split]
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "confs": [], "tps": [], "logarea": [],
                     "conf_all": [], "tp_all": []} for c in range(12)}
    det_rows = []  # for wilcoxon: (name, cls, conf, matched)
    per_image = {}
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        ims = []
        for r in chunk:
            p = os.path.join(COND, cond, split, r["name"] + ".jpg")
            im = cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1]
            ims.append(im.astype(np.float32) / 255.0)
        x = torch.from_numpy(np.ascontiguousarray(np.stack(ims).transpose(0, 3, 1, 2))).to(device)
        with torch.no_grad():
            pred = model(x)[0]
        pred = non_max_suppression(pred, 0.001, 0.5, max_det=300)
        for r, det in zip(chunk, pred):
            gts = load_gts(r["name"])
            if det is None:
                det = torch.zeros((0, 6))
            det = det.cpu().numpy()
            if len(det):
                det[:, :4] = det[:, :4].clip(0, 575)
            tp, mg = match_image(det, gts, 0.5)
            fpc = np.zeros(12, int); area_c = []
            # per-image FP counts at evaluation confidence
            for c in range(12):
                di = np.where(det[:, 5] == c)[0]
                gtc = int((gts[:, 0] == c).sum())
                per_class[c]["gt"] += gtc
                hi = di[det[di, 4] >= conf]
                ntp = int(tp[hi].sum())
                per_class[c]["tp"] += ntp
                per_class[c]["fp"] += int(len(hi) - ntp)
                fpc[c] = int(len(hi) - ntp)
                per_class[c]["fn"] += int(gtc - ntp)
                for i in di:
                    per_class[c]["conf_all"].append(float(det[i, 4]))
                    per_class[c]["tp_all"].append(int(tp[i]))
                    if det[i, 4] >= conf and tp[i]:
                        gi = mg[i]
                        pa = max((det[i, 2] - det[i, 0]) * (det[i, 3] - det[i, 1]), 1e-6)
                        ga = max((gts[gi, 3] - gts[gi, 1]) * (gts[gi, 4] - gts[gi, 2]), 1e-6)
                        per_class[c]["logarea"].append(float(np.log(pa / ga)))
                        area_c.append(float(np.log(pa / ga)))
                det_rows.append((r["name"], c, len(hi), ntp, int(len(hi) - ntp)))
            per_image[r["name"]] = {"fp_total": int(fpc.sum()), "fp": fpc.tolist(),
                                    "logarea": float(np.mean(area_c)) if area_c else None}
    out = {"cond": cond, "weights": weights, "split": split, "conf": conf, "classes": {}, "per_image": per_image}
    # FROC thresholds
    fps_grid = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    n_images = len(rows)
    for c in range(12):
        d = per_class[c]
        confs = np.array(d["conf_all"]); tps = np.array(d["tp_all"], bool)
        ap = ap_from_dets(confs, tps, d["gt"]) if d["gt"] else np.nan
        # FROC
        sens_at = {}
        if d["gt"] > 0 and len(confs):
            ths = np.unique(np.concatenate([[0.001], np.quantile(confs, np.linspace(0, 1, 200))]))
            sens, fppi = [], []
            for t in ths:
                sel = confs >= t
                sens.append(tps[sel].sum() / d["gt"])
                fppi.append((~tps[sel]).sum() / n_images)
            sens = np.array(sens); fppi = np.array(fppi)
            order = np.argsort(fppi)
            for f in fps_grid:
                sens_at[float(f)] = float(np.interp(f, fppi[order], sens[order]))
        prec = d["tp"] / max(d["tp"] + d["fp"], 1)
        rec = d["tp"] / max(d["gt"], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out["classes"][CLASS_NAMES[c]] = {
            "gt": int(d["gt"]), "tp": int(d["tp"]), "fp": int(d["fp"]), "fn": int(d["fn"]),
            "precision": float(prec), "recall": float(rec), "f1": float(f1), "ap50": ap,
            "median_log_area_ratio": float(np.median(d["logarea"])) if d["logarea"] else None,
            "median_area_ratio": float(np.exp(np.median(d["logarea"]))) if d["logarea"] else None,
            "froc_sens": sens_at,
        }
    # overall weighted
    tp = sum(v["tp"] for v in out["classes"].values()); fp = sum(v["fp"] for v in out["classes"].values())
    fn = sum(v["fn"] for v in out["classes"].values())
    ap_vals = [v["ap50"] for v in out["classes"].values() if v["ap50"] == v["ap50"]]
    out["overall"] = {"tp": tp, "fp": fp, "fn": fn,
                      "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                      "f1": 2 * tp / max(2 * tp + fp + fn, 1), "map50": float(np.mean(ap_vals)) if ap_vals else np.nan,
                      "fp_erosion": out["classes"]["erosion"]["fp"],
                      "fp_polyp": out["classes"]["polyp-like"]["fp"],
                      "fp_erythema": out["classes"]["erythema"]["fp"],
                      "fp_bleeding": out["classes"]["bleeding"]["fp"]}
    return out, det_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="")
    ap.add_argument("--runs", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    global CSV, ROW
    CSV = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    ROW = {r["name"]: r for r in CSV}
    os.makedirs(os.path.join(RES, "eval"), exist_ok=True)
    targets = []
    for cond in [c for c in args.conditions.split(",") if c]:
        for d in sorted(os.listdir(RUNS)):
            if d.startswith(cond + "_s") and os.path.exists(os.path.join(RUNS, d, "weights", "best.pt")):
                targets.append((cond, d, os.path.join(RUNS, d, "weights", "best.pt")))
    for r in [x for x in args.runs.split(",") if x]:
        cond = os.path.basename(r).split("_s")[0]
        w = os.path.join(r, "weights", "best.pt")
        if os.path.exists(w):
            targets.append((cond, os.path.basename(r), w))
    print("targets", len(targets), flush=True)
    rows_out = []
    for i, (cond, tag, w) in enumerate(targets):
        print("EVAL", tag, flush=True)
        res, det_rows = evaluate_model(w, cond, split=args.split, batch=args.batch, device=args.device)
        res["tag"] = tag
        json.dump(res, open(os.path.join(RES, "eval", f"{tag}.json"), "w"), indent=2)
        rows_out.append(res)
        torch.cuda.empty_cache()
    json.dump(rows_out, open(os.path.join(RES, "eval", f"summary_{args.split}.json"), "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
