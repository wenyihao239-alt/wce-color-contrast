import os, sys, csv, json, cv2, numpy as np
sys.path.insert(0, r"C:\wce\wce_experiment")
from wce.conditions import make_conditions, valid_mask
from wce.color import rgb_to_lab
from wce.metrics import masks, local_rms

RES = r"C:\wce\wce_experiment\results"
COND = r"C:\wce\wce_experiment\data\cond"
MANIFEST = os.path.join(RES, "manifest.csv")


def read(p):
    return cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float64)


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    val = [r for r in rows if r["split"] == "val" and int(r["n_boxes"]) > 0]
    rng = np.random.default_rng(42)
    val = [val[i] for i in rng.choice(len(val), min(120, len(val)), replace=False)]
    r3, r_med, e_clip = 64, 16, 1.0
    rms = {"E": [], "P2_0": [], "P2_1": [], "P1_0": [], "P1_1": [], "P1s_0": [], "P1s_1": []}
    for k, row in enumerate(val):
        n = row["name"]
        A = read(os.path.join(COND, "A", "val", n + ".jpg"))
        C = read(os.path.join(COND, "C", "val", n + ".jpg"))
        m = valid_mask(A)
        mucosa = masks(A)[1]
        outs1 = make_conditions(A, C, r3=r3, r_med=r_med, g_p2=1.0, g_p1=1.0, g_p1s=1.0, do_extra=True, e_clip=e_clip)
        outs0 = make_conditions(A, C, r3=r3, r_med=r_med, g_p2=0.0, g_p1=0.0, g_p1s=0.0, do_extra=True, e_clip=e_clip)
        rms["E"].append(local_rms(rgb_to_lab(outs1["E"])[..., 0], mucosa))
        for key, out in [("P2_1", outs1["P2"]), ("P1_1", outs1["P1"]), ("P1s_1", outs1["P1s"]),
                         ("P2_0", outs0["P2"]), ("P1_0", outs0["P1"]), ("P1s_0", outs0["P1s"])]:
            rms[key].append(local_rms(rgb_to_lab(out)[..., 0], mucosa))
        if k % 20 == 0:
            print(k, flush=True)
    target = float(np.mean(rms["E"]))

    def gain(r0, r1):
        r0 = float(np.mean(rms[r0])); r1 = float(np.mean(rms[r1]))
        if r1 <= r0:
            return 1.0
        g = np.sqrt(max(target ** 2 - r0 ** 2, 1e-6) / max(r1 ** 2 - r0 ** 2, 1e-6))
        return float(np.clip(g, 0.0, 8.0))

    g_p2 = gain("P2_0", "P2_1"); g_p1 = gain("P1_0", "P1_1"); g_p1s = gain("P1s_0", "P1s_1")
    params = {"stage": "extra", "r3": r3, "r_med": r_med, "e_clip": e_clip, "g_p2": g_p2, "g_p1": g_p1, "g_p1s": g_p1s,
              "target_rms": target, "rms": {k: float(np.mean(v)) for k, v in rms.items()},
              "n_images": len(val)}
    json.dump(params, open(os.path.join(RES, "params.json"), "w"), indent=2)
    print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
