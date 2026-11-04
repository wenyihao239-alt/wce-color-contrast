import os, csv, sys, shutil, yaml, argparse
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.data import CLASS_NAMES

RES = r"C:\wce\wce_experiment\results"
DATA = r"C:\wce\wce_experiment\data"
COND = os.path.join(DATA, "cond")
RAW = r"C:\wce\data\raw"
MANIFEST = os.path.join(RES, "manifest.csv")
LABELS = os.path.join(DATA, "labels")


def link(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def label_src(r):
    n = r["name"]
    if r["source"] == "origd":
        sp = "training" if r["official_split"] == "train" else "validation"
        return os.path.join(RAW, "origd", "labels", sp, n + ".txt")
    return os.path.join(RAW, "seeai_full", "SEE_AI_project_all_txt", "SEE_AI_project_all_txt", n + ".txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="A,C,M1,M2")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    for r in rows:
        dst = os.path.join(LABELS, r["split"], r["name"] + ".txt")
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            src = label_src(r)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                open(dst, "w").close()
    print("labels done", flush=True)
    for cond in [c for c in args.conditions.split(",") if c]:
        root = os.path.join(DATA, "yolo", cond)
        missing = 0
        for r in rows:
            src = os.path.join(COND, cond, r["split"], r["name"] + ".jpg")
            if not os.path.exists(src):
                missing += 1
                continue
            link(src, os.path.join(root, "images", r["split"], r["name"] + ".jpg"))
            link(os.path.join(LABELS, r["split"], r["name"] + ".txt"),
                 os.path.join(root, "labels", r["split"], r["name"] + ".txt"))
        y = {"path": root.replace("\\", "/"), "train": "images/train", "val": "images/val",
             "test": "images/test", "nc": 12, "names": CLASS_NAMES}
        os.makedirs(os.path.join(DATA, "yaml"), exist_ok=True)
        with open(os.path.join(DATA, "yaml", cond + ".yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(y, f, allow_unicode=True, sort_keys=False)
        print("prepared", cond, "missing", missing, flush=True)


if __name__ == "__main__":
    main()
