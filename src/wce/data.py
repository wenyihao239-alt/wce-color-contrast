"""Manifest construction, perceptual clustering and leakage-safe split."""
import os, csv, numpy as np, cv2, json
from collections import Counter, defaultdict

RAW = r"C:\wce\data\raw"
FULL_IMG = os.path.join(RAW, "seeai_full", "SEE_AI_project_all_images", "SEE_AI_project_all_images")
FULL_TXT = os.path.join(RAW, "seeai_full", "SEE_AI_project_all_txt", "SEE_AI_project_all_txt")
ORIG = os.path.join(RAW, "origd")
BG = cv2.imread(r"C:\wce\WCE2024\Retinanet\theBG.jpeg")

CLASS_NAMES = ['angiodysplasia', 'erosion', 'stenosis', 'lymphangiectasia', 'lymph-follicle', 'SMT',
               'polyp-like', 'bleeding', 'diverticulum', 'erythema', 'foreign-body', 'vein']


def read_label(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        p = line.split()
        if len(p) == 5:
            out.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def build_manifest():
    orig_split = {}
    for sp, ys in [("training", "train"), ("validation", "val")]:
        d = os.path.join(ORIG, "images", sp)
        for f in os.listdir(d):
            orig_split[f[:-4]] = (sp, ys)
    full_names = sorted(f[:-4] for f in os.listdir(FULL_IMG) if f.endswith(".jpg"))
    rows = []
    for n in full_names:
        if n in orig_split:
            src = "origd"
            sp, off = orig_split[n]
            lab = read_label(os.path.join(ORIG, "labels", sp, n + ".txt"))
        else:
            src = "full"
            lab = read_label(os.path.join(FULL_TXT, n + ".txt"))
            off = ""
        rows.append({"name": n, "source": src, "official_split": off, "n_boxes": len(lab),
                     "classes": ";".join(str(x[0]) for x in lab)})
    # sanity
    assert len(rows) == 18481, len(rows)
    assert sum(r["n_boxes"] for r in rows) == 23033, sum(r["n_boxes"] for r in rows)
    return rows


def dhash(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits(g[:, 1:] > g[:, :-1]).view(np.uint8)


def chist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256]).ravel().astype(np.float32)
    h /= (np.linalg.norm(h) + 1e-9)  # L2 normalisation -> dot product is cosine similarity
    return h


def compute_hashes(rows):
    cache = r"C:\wce\wce_experiment\results\hashes.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        if len(z["hashes"]) == len(rows):
            print("loaded hash cache", flush=True)
            return z["hashes"].view(np.uint64).ravel(), z["hists"]
    hashes = np.zeros((len(rows), 8), np.uint8)
    hists = np.zeros((len(rows), 256), np.float32)
    for i, r in enumerate(rows):
        p = os.path.join(ORIG, "images", "training" if r["official_split"] == "train" else "validation", r["name"] + ".jpg") \
            if r["source"] == "origd" else os.path.join(FULL_IMG, r["name"] + ".jpg")
        im = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4)
        if im is None:
            raise RuntimeError(p)
        hashes[i] = dhash(im)
        hists[i] = chist(im)
        if i % 2000 == 0:
            print("hash", i, flush=True)
    # pack 8 bytes into uint64 for fast hamming
    np.savez_compressed(cache, hashes=hashes, hists=hists)
    return hashes.view(np.uint64).ravel(), hists


POP = np.array([bin(i).count("1") for i in range(65536)], np.uint8)


def hamming_matrix_chunk(h, idx):
    a = h[idx][:, None]
    x = np.bitwise_xor(a, h[None, :]).view(np.uint16).reshape(len(idx), -1, 4)
    return POP[x].sum(2, dtype=np.uint8)


def cluster(rows, h, hists, ham_thr=8, cos_thr=0.98, max_gap=200):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(rows)
    ri, ci = [], []
    for start in range(0, n, 512):
        idx = np.arange(start, min(start + 512, n))
        ham = hamming_matrix_chunk(h, idx)
        cos = hists[idx] @ hists.T
        jj = np.arange(n)[None, :]
        mask = (ham <= ham_thr) & (cos >= cos_thr) & (jj > idx[:, None]) & (jj <= idx[:, None] + max_gap)
        ii, jv = np.where(mask)
        ri.append(idx[ii].astype(np.int32)); ci.append(jv.astype(np.int32))
        if start % 5120 == 0:
            print("cluster chunk", start, "edges", sum(len(x) for x in ri), flush=True)
    ri = np.concatenate(ri); ci = np.concatenate(ci)
    print("total edges", len(ri), flush=True)
    g = coo_matrix((np.ones(len(ri), np.uint8), (ri, ci)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab


def assign_split(rows, lab, seed=42):
    n = len(rows)
    rng = np.random.default_rng(seed)
    clusters = defaultdict(list)
    for i, c in enumerate(lab):
        clusters[c].append(i)
    keys = list(clusters.keys())
    rng.shuffle(keys)
    # greedy: assign each cluster to split with largest deficit vs target
    target = np.array([0.70, 0.15, 0.15])
    counts = np.zeros(3)
    total = n
    split = np.empty(n, dtype=object)
    # sort clusters by size descending, keep random tie order
    keys.sort(key=lambda k: -len(clusters[k]))
    for k in keys:
        idxs = clusters[k]
        deficits = target * total - counts
        # prefer split with largest relative deficit, but never let test/val overflow
        s = int(np.argmax(deficits / target))
        if counts[s] + len(idxs) > target[s] * total * 1.15 and s != 0:
            s = int(np.argmin(counts / np.maximum(target, 1e-9)))
        counts[s] += len(idxs)
        split[idxs] = ["train", "val", "test"][s]
    return split, counts


def ensure_classes(rows, lab, split):
    # ensure every class has boxes in val and test; move clusters if missing
    cls_of = [set(int(x) for x in r["classes"].split(";") if x != "") for r in rows]
    for s in ["val", "test"]:
        for c in range(12):
            present = any(split[i] == s and c in cls_of[i] and rows[i]["n_boxes"] > 0 for i in range(len(rows)))
            if not present:
                cand = [i for i in range(len(rows)) if split[i] == "train" and c in cls_of[i]]
                if not cand:
                    print("WARNING class", c, "missing everywhere")
                    continue
                # move the whole cluster of a candidate
                cl = lab[cand[0]]
                members = np.where(lab == cl)[0]
                split[members] = s
                print("moved cluster for class", c, "->", s)
    return split


def main():
    out = r"C:\wce\wce_experiment\results"
    rows = build_manifest()
    print("manifest", len(rows), "normal", sum(r["n_boxes"] == 0 for r in rows))
    h, hists = compute_hashes(rows)
    print("hashes done")
    lab = cluster(rows, h, hists)
    print("n clusters", len(set(lab)), "max size", max(Counter(lab).values()))
    split, counts = assign_split(rows, lab)
    split = ensure_classes(rows, lab, split)
    for i, r in enumerate(rows):
        r["cluster"] = int(lab[i]); r["split"] = split[i]
    with open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "source", "official_split", "n_boxes", "classes", "cluster", "split"])
        w.writeheader(); w.writerows(rows)
    # stats
    stats = {"n": len(rows), "clusters": int(len(set(lab))), "splits": {}}
    for s in ["train", "val", "test"]:
        ix = [i for i in range(len(rows)) if split[i] == s]
        cc = Counter()
        for i in ix:
            for c in rows[i]["classes"].split(";"):
                if c != "":
                    cc[int(c)] += 1
        stats["splits"][s] = {"images": len(ix), "boxes": int(sum(rows[i]["n_boxes"] for i in ix)),
                              "classes": {CLASS_NAMES[k]: int(v) for k, v in sorted(cc.items())},
                              "normal": int(sum(rows[i]["n_boxes"] == 0 for i in ix))}
    json.dump(stats, open(os.path.join(out, "split_stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
