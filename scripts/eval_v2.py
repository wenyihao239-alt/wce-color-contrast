"""Audited evaluator: low-threshold inference, AP50/AP50-95, PR/FROC, per-image FP."""
import os, sys, csv, json, argparse, cv2, numpy as np, torch
sys.path.insert(0, r"C:\wce\WCE2024\Yolov5\yolov5")
from models.experimental import attempt_load
from utils.general import non_max_suppression
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.data import CLASS_NAMES

RES = r"C:\wce\wce_experiment\results"
COND = r"C:\wce\wce_experiment\data\cond"
MANIFEST = os.path.join(RES, "manifest.csv")
OUT = os.path.join(RES, "v2_audit")
os.makedirs(OUT, exist_ok=True)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match(dets, gts, iou_thr):
    tp = np.zeros(len(dets), bool); mg = -np.ones(len(dets), int)
    for c in range(12):
        di = np.where(dets[:, 5] == c)[0]
        gi = np.where(gts[:, 0] == c)[0]
        if len(di) == 0 or len(gi) == 0:
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


def ap_from(confs, tps, npos, npoints=101):
    if npos == 0:
        return float("nan")
    if len(confs) == 0:
        return 0.0
    order = np.argsort(-np.asarray(confs))
    tps = np.asarray(tps, bool)[order]
    ctp = np.cumsum(tps); cfp = np.cumsum(~tps)
    rec = ctp / npos; prec = ctp / (ctp + cfp + 1e-9)
    mrec = np.linspace(0, 1, npoints)
    return float(np.interp(mrec, rec, prec).mean())


def pr_curve(confs, tps, npos, npoints=200):
    if npos == 0 or len(confs) == 0:
        return np.zeros(npoints), np.zeros(npoints)
    order = np.argsort(-np.asarray(confs))
    tps = np.asarray(tps, bool)[order]; confs = np.asarray(confs)[order]
    ctp = np.cumsum(tps); cfp = np.cumsum(~tps)
    rec = ctp / npos; prec = ctp / (ctp + cfp + 1e-9)
    grid = np.linspace(0, 1, npoints)
    return grid, np.interp(grid, rec, prec)


def load_gts(row):
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
                c, cx, cy, w, h = int(q[0]), *map(float, q[1:])
                out.append([c, (cx - w / 2) * 576, (cy - h / 2) * 576,
                            (cx + w / 2) * 576, (cy + h / 2) * 576])
    return np.array(out, float) if out else np.zeros((0, 5))


def evaluate(weights, tag, cond, split, batch=16, device="cuda:0", conf_work=0.24):
    model = attempt_load(weights, device=device); model.eval().float()
    rows = [r for r in CSV if r["split"] == split]
    records = []
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
            det = det.cpu().numpy() if det is not None else np.zeros((0, 6))
            if len(det):
                det[:, :4] = det[:, :4].clip(0, 575)
            records.append((r["name"], det, load_gts(r)))
    ious = np.arange(0.5, 1.0, 0.05)
    ap_by_iou = {float(i): {} for i in ious}
    cls_data = {c: {"gt": 0, "conf": [], "tp05": []} for c in range(12)}
    per_image = {}
    for name, det, gts in records:
        fpc = np.zeros(12, int); area_c = []; area_logs = {c: [] for c in range(12)}
        for c in range(12):
            cls_data[c]["gt"] += int((gts[:, 0] == c).sum())
        for iou in ious:
            tp, mg = match(det, gts, float(iou))
            for c in range(12):
                di = np.where(det[:, 5] == c)[0]
                if float(iou) == 0.5:
                    cls_data[c]["conf"].extend(det[di, 4].tolist())
                    cls_data[c]["tp05"].extend(tp[di].tolist())
                ap_by_iou[float(iou)].setdefault(c, ([], []))
                ap_by_iou[float(iou)][c][0].extend(det[di, 4].tolist())
                ap_by_iou[float(iou)][c][1].extend(tp[di].tolist())
        tp05, mg05 = match(det, gts, 0.5)
        for c in range(12):
            di = np.where(det[:, 5] == c)[0]
            hi = di[det[di, 4] >= conf_work]
            ntp = int(tp05[hi].sum())
            fpc[c] = int(len(hi) - ntp)
            for i in hi:
                if tp05[i]:
                    gi = mg05[i]
                    pa = max((det[i, 2] - det[i, 0]) * (det[i, 3] - det[i, 1]), 1e-6)
                    ga = max((gts[gi, 3] - gts[gi, 1]) * (gts[gi, 4] - gts[gi, 2]), 1e-6)
                    area_c.append(float(np.log(pa / ga)))
                    area_logs[c].append(float(np.log(pa / ga)))
        per_image[name] = {"fp_total": int(fpc.sum()), "fp": fpc.tolist(),
                           "logarea": float(np.mean(area_c)) if area_c else None,
                           "area_logs": {str(k): v for k, v in area_logs.items()},
                           "n_boxes": int(len(gts))}
    out = {"tag": tag, "cond": cond, "split": split, "weights": weights,
           "conf_work": conf_work, "classes": {}, "per_image": per_image}
    for c in range(12):
        npos = cls_data[c]["gt"]
        confs = cls_data[c]["conf"]; tps = cls_data[c]["tp05"]
        aps = {}
        for iou in ious:
            aps[float(iou)] = ap_from(ap_by_iou[float(iou)][c][0], ap_by_iou[float(iou)][c][1], npos)
        ap50 = aps[0.5]
        ap5095 = float(np.nanmean([v for v in aps.values()]))
        di = np.array(confs); ti = np.array(tps, bool)
        work = di >= conf_work
        tp_w = int(ti[work].sum()); fp_w = int(work.sum() - tp_w)
        prec = tp_w / max(tp_w + fp_w, 1); rec = tp_w / max(npos, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        areas = [per_image[n]["logarea"] for n, d, g in records if per_image[n]["logarea"] is not None]
        out["classes"][CLASS_NAMES[c]] = {
            "gt": int(npos), "tp": tp_w, "fp": fp_w, "fn": int(npos - tp_w),
            "precision": prec, "recall": rec, "f1": f1, "ap50": ap50, "ap50_95": ap5095,
            "median_log_area_ratio_at_work": None,
        }
        # per-class matched area at work point
        la = []
        for n in per_image:
            la.extend(per_image[n]["area_logs"][str(c)])
        out["classes"][CLASS_NAMES[c]]["median_area_ratio"] = float(np.exp(np.median(la))) if la else None
        # PR curve
        grid, prec_curve = pr_curve(di, ti, npos)
        out["classes"][CLASS_NAMES[c]]["pr_recall"] = grid.tolist()
        out["classes"][CLASS_NAMES[c]]["pr_precision"] = prec_curve.tolist()
    tp = sum(v["tp"] for v in out["classes"].values())
    fp = sum(v["fp"] for v in out["classes"].values())
    fn = sum(v["fn"] for v in out["classes"].values())
    aps50 = [v["ap50"] for v in out["classes"].values() if v["ap50"] == v["ap50"]]
    aps = [v["ap50_95"] for v in out["classes"].values() if v["ap50_95"] == v["ap50_95"]]
    out["overall"] = {"tp": tp, "fp": fp, "fn": fn,
                      "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                      "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                      "map50": float(np.mean(aps50)), "map50_95": float(np.mean(aps)),
                      "fp_normal": int(sum(d["fp_total"] for n, d in per_image.items() if d["n_boxes"] == 0)),
                      "n_normal": int(sum(1 for d in per_image.values() if d["n_boxes"] == 0)),
                      "fp_abnormal": int(sum(d["fp_total"] for d in per_image.values() if d["n_boxes"] > 0)),
                      "n_abnormal": int(sum(1 for d in per_image.values() if d["n_boxes"] > 0))}
    json.dump(out, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--last", action="store_true")
    args = ap.parse_args()
    global CSV
    CSV = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    wn = "last.pt" if args.last else "best.pt"
    sfx = "_last" if args.last else ""
    targets = []
    for t in [x for x in args.tags.split(",") if x]:
        cond = t.split("_s")[0]
        targets.append((t + sfx, cond, os.path.join(r"C:\wce\wce_experiment\runs\controlled", t, "weights", wn)))
    if not targets:
        for c in ["A", "C", "M1", "M2", "P0", "P2", "E"]:
            if os.path.exists(os.path.join(r"C:\wce\wce_experiment\runs\controlled", f"{c}_s42", "weights", wn)):
                targets.append((f"{c}_s42{sfx}", c, os.path.join(r"C:\wce\wce_experiment\runs\controlled", f"{c}_s42", "weights", wn)))
    for tag, cond, w in targets:
        print("EVAL", tag, flush=True)
        evaluate(w, tag, cond, args.split, batch=args.batch)


if __name__ == "__main__":
    main()
