import os, json, numpy as np
from scipy.stats import wilcoxon

RES = r"C:\wce\wce_experiment\results\v2_eval"
PAIRS = [("S", "O"), ("H", "S"), ("H", "E"), ("E", "S"), ("L", "O"), ("X", "O")]
TAGS = {"O": "O original", "S": "S calibrated", "H": "H joint control", "E": "E calibrated+CLAHE",
        "L": "L lightness-only", "X": "X chroma-only"}


def load(cond, seed=42):
    p = os.path.join(RES, f"{cond}_s{seed}_last.json")
    return json.load(open(p)) if os.path.exists(p) else None


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); out = np.ones(m); prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i]); out[i] = max(prev, val); prev = out[i]
    return out


def main():
    data = {c: load(c) for c in ["O", "S", "H", "E", "L", "X"]}
    rows = []
    for a, b in PAIRS:
        da, db = data[a], data[b]
        names = sorted(set(da["per_image"]) & set(db["per_image"]))
        fp_a = np.array([da["per_image"][n]["fp_total"] for n in names], float)
        fp_b = np.array([db["per_image"][n]["fp_total"] for n in names], float)
        la_a, la_b = [], []
        for n in names:
            va = da["per_image"][n]["area_logs"]
            vb = db["per_image"][n]["area_logs"]
            va = [x for xs in va.values() for x in xs]
            vb = [x for xs in vb.values() for x in xs]
            if va and vb:
                la_a.append(np.mean(va)); la_b.append(np.mean(vb))
        la_a = np.array(la_a); la_b = np.array(la_b)
        try:
            p_fp = wilcoxon(fp_a, fp_b, zero_method="wilcox").pvalue
        except Exception:
            p_fp = 1.0
        try:
            p_la = wilcoxon(la_a, la_b, zero_method="wilcox").pvalue
        except Exception:
            p_la = 1.0
        rows.append({"a": a, "b": b, "fp_a": float(fp_a.mean()), "fp_b": float(fp_b.mean()),
                     "fp_diff": float((fp_a - fp_b).mean()), "p_fp": float(p_fp),
                     "area_a": float(np.exp(la_a.mean())) if len(la_a) else None,
                     "area_b": float(np.exp(la_b.mean())) if len(la_b) else None,
                     "p_area": float(p_la), "n": len(names)})
    pf = holm([r["p_fp"] for r in rows]); pa = holm([r["p_area"] for r in rows])
    for r, x, y in zip(rows, pf, pa):
        r["p_fp_holm"] = float(x); r["p_area_holm"] = float(y)
    json.dump(rows, open(os.path.join(RES, "stats_v2.json"), "w"), indent=2)
    print("comparison  dFP  p_FP  p_holm  areaA areaB p_area")
    for r in rows:
        print(f"{r['a']} vs {r['b']:2s} {r['fp_diff']:+.3f} {r['p_fp']:.2e} {r['p_fp_holm']:.2e} "
              f"{(r['area_a'] or 0):.3f} {(r['area_b'] or 0):.3f} {r['p_area']:.2e}")
    print("seed means O:", data["O"]["overall"]["map50"], load("O", 2025)["overall"]["map50"])
    print("seed means S:", data["S"]["overall"]["map50"], load("S", 2025)["overall"]["map50"])


if __name__ == "__main__":
    main()
