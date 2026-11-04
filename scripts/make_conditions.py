import os, sys, csv, json, argparse, cv2, numpy as np, shutil
from multiprocessing import Pool
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.conditions import (RAW, RES, COND, FULL_IMG, BG, M_CCC, B_CCC, to_rgb, save_rgb,
                            valid_mask, blend, make_conditions)
from wce.color import apply_affine

MANIFEST = os.path.join(RES, "manifest.csv")
ORIG = os.path.join(RAW, "origd")
CCCD = os.path.join(RAW, "cccd")

PARAMS = {}
if os.path.exists(os.path.join(RES, "params.json")):
    PARAMS = json.load(open(os.path.join(RES, "params.json")))


def orig_path(row, kind, name):
    sp = "training" if row["official_split"] == "train" else "validation"
    return os.path.join(RAW, kind, "images", sp, name + ".jpg")


def process(row):
    name, split, src = row["name"], row["split"], row["source"]
    stage = PARAMS.get("stage", "base")
    light = PARAMS.get("light", True)
    keys = ["M1", "M2", "M3"] if stage == "base" else (["P0", "P2", "E"] if light else
           ["M1", "M2", "M3", "P0", "P2", "P1", "P1s", "E", "D"])
    ap = os.path.join(COND, "A", split, name + ".jpg")
    cp = os.path.join(COND, "C", split, name + ".jpg")
    if os.path.exists(ap) and os.path.exists(cp) and all(
            os.path.exists(os.path.join(COND, k, split, name + ".jpg")) for k in keys):
        return name
    if src == "origd":
        A = to_rgb(cv2.imread(orig_path(row, "origd", name), cv2.IMREAD_COLOR))
        Cpath = orig_path(row, "cccd", name)
        C = to_rgb(cv2.imread(Cpath, cv2.IMREAD_COLOR))
    else:
        full = cv2.imread(os.path.join(FULL_IMG, name + ".jpg"), cv2.IMREAD_COLOR).astype(np.float64)
        A = np.clip(np.round(full * BG / 255.0), 0, 255)[:, :, ::-1]
        C = None

    done = []
    # A condition
    ap = os.path.join(COND, "A", split, name + ".jpg")
    if not os.path.exists(ap):
        os.makedirs(os.path.dirname(ap), exist_ok=True)
        if src == "origd":
            shutil.copy2(orig_path(row, "origd", name), ap)
        else:
            save_rgb(ap, A)
    # C condition
    cp = os.path.join(COND, "C", split, name + ".jpg")
    if not os.path.exists(cp):
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        if src == "origd":
            shutil.copy2(Cpath, cp)
        else:
            save_rgb(cp, np.clip(apply_affine(A, M_CCC, B_CCC), 0, 255))

    stage = PARAMS.get("stage", "base")
    outs = make_conditions(A, C, r3=PARAMS.get("r3", 64), r_med=PARAMS.get("r_med", 16),
                           g_p2=PARAMS.get("g_p2", 1.0), g_p1=PARAMS.get("g_p1", 1.0),
                           g_p1s=PARAMS.get("g_p1s", 1.0), do_extra=(stage == "extra"), e_clip=PARAMS.get("e_clip", 1.0),
                           light=light)
    keys = ["M1", "M2", "M3"] if stage == "base" else (["P0", "P2", "E"] if light else
           ["M1", "M2", "M3", "P0", "P2", "P1", "P1s", "E", "D"])
    for k in keys:
        p = os.path.join(COND, k, split, name + ".jpg")
        if not os.path.exists(p):
            save_rgb(p, outs[k])
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["base", "extra"], default="base")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--threads", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    PARAMS["stage"] = args.stage
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]
    print("rows", len(rows), "stage", args.stage, "workers", args.workers, "threads", args.threads, flush=True)
    if args.threads:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, n in enumerate(ex.map(process, rows)):
                if i % 500 == 0:
                    print("done", i, flush=True)
    else:
        with Pool(args.workers) as pool:
            for i, n in enumerate(pool.imap_unordered(process, rows, chunksize=16)):
                if i % 500 == 0:
                    print("done", i, flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
