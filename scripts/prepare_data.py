import os, csv, json, shutil, yaml, sys
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.data import CLASS_NAMES

RES = r"C:\wce\wce_experiment\results"
DATA = r"C:\wce\wce_experiment\data"
COND = os.path.join(DATA, "cond")
RAW = r"C:\wce\data\raw"
MANIFEST = os.path.join(RES, "manifest.csv")
LABELS = os.path.join(DATA, "labels")


def label_src(row):
    n = row["name"]
    if row["source"] == "origd":
        sp = "training" if row["official_split"] == "train" else "validation"
        return os.path.join(RAW, "origd", "labels", sp, n + ".txt")
    return os.path.join(RAW, "seeai_full", "SEE_AI_project_all_txt", "SEE_AI_project_all_txt", n + ".txt")


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    # master labels
    for r in rows:
        dst = os.path.join(LABELS, r["split"], r["name"] + ".txt")
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            src = label_src(r)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                open(dst, "w").close()
    print("labels ready", flush=True)

    conds = [d for d in os.listdir(COND) if os.path.isdir(os.path.join(COND, d))]
    for c in conds:
        base = os.path.join(COND, c)
        for sp in ["train", "val", "test"]:
            src = os.path.join(base, sp)
            if os.path.isdir(src) and not os.path.isdir(os.path.join(base, "images", sp)):
                os.makedirs(os.path.join(base, "images"), exist_ok=True)
                os.rename(src, os.path.join(base, "images", sp))
            for n in os.listdir(os.path.join(LABELS, sp)):
                link_or_copy(os.path.join(LABELS, sp, n), os.path.join(base, "labels", sp, n))
        y = {"path": base.replace("\\", "/"), "train": "images/train", "val": "images/val",
             "test": "images/test", "nc": 12, "names": CLASS_NAMES}
        os.makedirs(os.path.join(DATA, "yaml"), exist_ok=True)
        with open(os.path.join(DATA, "yaml", c + ".yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(y, f, allow_unicode=True, sort_keys=False)
        print("prepared", c, flush=True)

    # official anchor yamls
    for name, path in [("anchor_origd", os.path.join(RAW, "origd")), ("anchor_cccd", os.path.join(RAW, "cccd")),
                       ("anchor_ccd", os.path.join(RAW, "ccd"))]:
        y = {"path": path.replace("\\", "/"), "train": "images/training", "val": "images/validation",
             "nc": 12, "names": CLASS_NAMES}
        with open(os.path.join(DATA, "yaml", name + ".yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(y, f, allow_unicode=True, sort_keys=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
