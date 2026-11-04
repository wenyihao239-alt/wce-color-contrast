import os, json, itertools, numpy as np
from scipy.stats import wilcoxon

RES = r"C:\wce\wce_experiment\results\eval"
PAIRS = [("C", "A"), ("M1", "C"), ("M2", "C"), ("P0", "C"), ("P2", "E"), ("E", "C")]
LABELS = {"C": "CCC-clipped", "A": "Original", "M1": "CCC-lightness", "M2": "CCC-chroma",
          "P0": "Detail transfer", "P2": "Proposed P2", "E": "CCC+CLAHE"}


def load(cond):
    p = os.path.join(RES, f"{cond}_s42.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals); out = np.ones(m)
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        out[i] = max(prev, val); prev = out[i]
    return out


def main():
    data = {c: load(c) for c in set(sum(([a, b] for a, b in PAIRS), []))}
    for c, d in data.items():
        if d is None:
            print("missing eval for", c)
    rows = []
    for a, b in PAIRS:
        da, db = data.get(a), data.get(b)
        if not da or not db:
            continue
        names = sorted(set(da["per_image"]) & set(db["per_image"]))
        fp_a = np.array([da["per_image"][n]["fp_total"] for n in names], float)
        fp_b = np.array([db["per_image"][n]["fp_total"] for n in names], float)
        la_a = np.array([da["per_image"][n]["logarea"] for n in names if da["per_image"][n]["logarea"] is not None
                         and db["per_image"][n]["logarea"] is not None], float)
        la_b = np.array([db["per_image"][n]["logarea"] for n in names if da["per_image"][n]["logarea"] is not None
                         and db["per_image"][n]["logarea"] is not None], float)
        try:
            p_fp = wilcoxon(fp_a, fp_b, zero_method="wilcox").pvalue
        except Exception:
            p_fp = 1.0
        try:
            p_la = wilcoxon(la_a, la_b, zero_method="wilcox").pvalue
        except Exception:
            p_la = 1.0
        rows.append({"a": a, "b": b, "n": len(names), "n_area": len(la_a),
                     "fp_a": float(fp_a.mean()), "fp_b": float(fp_b.mean()),
                     "fp_diff": float((fp_a - fp_b).mean()), "p_fp": float(p_fp),
                     "logarea_a": float(la_a.mean()) if len(la_a) else None,
                     "logarea_b": float(la_b.mean()) if len(la_b) else None,
                     "area_ratio_a": float(np.exp(la_a.mean())) if len(la_a) else None,
                     "area_ratio_b": float(np.exp(la_b.mean())) if len(la_b) else None,
                     "p_area": float(p_la)})
    pf = holm([r["p_fp"] for r in rows]); pa = holm([r["p_area"] for r in rows])
    for r, f, g in zip(rows, pf, pa):
        r["p_fp_holm"] = float(f); r["p_area_holm"] = float(g)
    json.dump(rows, open(os.path.join(RES, "stats.json"), "w"), indent=2)
    print(f"{'comparison':22s} {'dFP':>7s} {'p_FP':>9s} {'p_holm':>9s} {'areaA':>6s} {'areaB':>6s} {'p_area':>9s}")
    for r in rows:
        print(f"{r['a']+' vs '+r['b']:22s} {r['fp_diff']:7.3f} {r['p_fp']:9.2e} {r['p_fp_holm']:9.2e} "
              f"{(r['area_ratio_a'] or 0):6.3f} {(r['area_ratio_b'] or 0):6.3f} {r['p_area']:9.2e}")


if __name__ == "__main__":
    main()
