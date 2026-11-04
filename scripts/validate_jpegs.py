import os, sys, json
BASE = r"C:\wce\wce_experiment\data\cond"
bad = []
for root, dirs, files in os.walk(BASE):
    for f in files:
        if not f.lower().endswith(".jpg"):
            continue
        p = os.path.join(root, f)
        try:
            with open(p, "rb") as fh:
                data = fh.read()
            if len(data) < 1000 or not data.rstrip().endswith(b"\xff\xd9"):
                bad.append(p)
        except Exception:
            bad.append(p)
for p in bad:
    try:
        os.remove(p)
    except Exception:
        pass
print("bad_removed", len(bad))
