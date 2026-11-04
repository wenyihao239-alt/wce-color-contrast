import os, sys, csv, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\wce\wce_experiment\scripts")
from eval_v2 import match, ap_from, iou_matrix

RES = r"C:\wce\wce_experiment\results"
OUT = os.path.join(RES, "v2_audit")
os.makedirs(OUT, exist_ok=True)


def case(name, dets, gts, expect_tp, expect_fp):
    tp, mg = match(np.array(dets, float) if dets else np.zeros((0, 6)), np.array(gts, float) if gts else np.zeros((0, 5)), 0.5)
    ok = int(tp.sum()) == expect_tp and int((~tp).sum()) == expect_fp
    return {"case": name, "tp": int(tp.sum()), "fp": int((~tp).sum()), "pass": bool(ok)}


def main():
    cases = []
    cases.append(case("exact_single", [[10, 10, 50, 50, 0.9, 0]], [[0, 10, 10, 50, 50]], 1, 0))
    cases.append(case("wrong_class", [[10, 10, 50, 50, 0.9, 1]], [[0, 10, 10, 50, 50]], 0, 1))
    cases.append(case("duplicate_preds", [[10, 10, 50, 50, 0.9, 0], [11, 11, 49, 49, 0.8, 0]], [[0, 10, 10, 50, 50]], 1, 1))
    cases.append(case("gt_no_pred", [], [[0, 10, 10, 50, 50]], 0, 0))
    cases.append(case("pred_no_gt", [[10, 10, 50, 50, 0.9, 0]], [], 0, 1))
    cases.append(case("empty", [], [], 0, 0))
    ap = ap_from([0.9, 0.5], [True, False], 1)
    cases.append({"case": "ap_simple", "ap": ap, "pass": abs(ap - 1.0) < 1e-9})

    # class / label audit
    y = open(r"C:\wce\WCE2024\Yolov5\dataset.yaml", encoding="utf-8").read()
    names = ['angiodysplasia', 'erosion', 'stenosis', 'lymphangiectasia', 'lymph-follicle', 'SMT',
             'polyp-like', 'bleeding', 'diverticulum', 'erythema', 'foreign-body', 'vein']
    rows = list(csv.DictReader(open(os.path.join(RES, "manifest.csv"), encoding="utf-8")))
    cls_seen = set(); bad_label = 0; normal_nonempty = 0; abnormal_empty = 0
    for r in rows[:3000]:
        path = os.path.join(RES, ".tmp_none")
        if r["source"] == "origd":
            sp = "training" if r["official_split"] == "train" else "validation"
            p = os.path.join(r"C:\wce\data\raw\origd", "labels", sp, r["name"] + ".txt")
        else:
            p = os.path.join(r"C:\wce\data\raw\seeai_full", "SEE_AI_project_all_txt", "SEE_AI_project_all_txt", r["name"] + ".txt")
        if not os.path.exists(p):
            continue
        lines = [x for x in open(p) if x.strip()]
        if int(r["n_boxes"]) == 0 and lines:
            normal_nonempty += 1
        if int(r["n_boxes"]) > 0 and not lines:
            abnormal_empty += 1
        for line in lines:
            q = line.split()
            if len(q) != 5 or not (0 <= int(q[0]) <= 11):
                bad_label += 1
            else:
                cls_seen.add(int(q[0]))
    label_audit = {"classes_seen": sorted(cls_seen), "bad_label": bad_label,
                   "normal_nonempty": normal_nonempty, "abnormal_empty": abnormal_empty,
                   "class_names": names, "background_comment_present": "0 index is reserved for background" in y}

    # checkpoint / threshold audit
    thr = {"val_py_default_conf": 0.2499, "train_validate_overrides_conf": False,
           "source": "Yolov5/yolov5/val.py run(conf_thres=0.2499); train.py validate.run(...) omits conf_thres"}
    ck = []
    for p in sorted(glob.glob(os.path.join(r"C:\wce\wce_experiment\runs\controlled", "*", "results.csv"))):
        df = pd.read_csv(p); df.columns = [c.strip() for c in df.columns]
        tag = os.path.basename(os.path.dirname(p))
        best_ep = int(df.loc[df["metrics/mAP_0.5:0.95"].idxmax(), "epoch"])
        ck.append({"tag": tag, "epochs_logged": int(len(df)), "best_epoch_by_high_thr_val": best_ep,
                   "last_epoch": int(df.iloc[-1]["epoch"])})
    cmp = []
    for c in ["A", "C", "M1", "M2", "P0", "P2", "E"]:
        b = json.load(open(os.path.join(OUT, f"{c}_s42.json")))
        l = json.load(open(os.path.join(OUT, f"{c}_s42_last.json")))
        cmp.append({"cond": c, "best_ap50": b["overall"]["map50"], "best_ap5095": b["overall"]["map50_95"],
                    "last_ap50": l["overall"]["map50"], "last_ap5095": l["overall"]["map50_95"],
                    "best_fp": b["overall"]["fp"], "last_fp": l["overall"]["fp"],
                    "ranking_change": None})

    out = {"unit_tests": cases, "label_audit": label_audit, "threshold_audit": thr,
           "checkpoint_audit": ck, "best_vs_last": cmp}
    json.dump(out, open(os.path.join(OUT, "audit_report.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
