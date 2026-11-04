import zipfile, os, sys, time

BASE = r"C:\Users\YOLO\Documents\胶囊内镜\data\downloads"
OUT = r"C:\Users\YOLO\Documents\胶囊内镜\data\raw"
jobs = [
    ("archive.zip", "origd", None),
    ("archive (1).zip", "ccd", "data"),
    ("archive (2).zip", "cccd", "data"),
]

for zname, outname, strip in jobs:
    zpath = os.path.join(BASE, zname)
    dst = os.path.join(OUT, outname)
    if os.path.isdir(dst) and len(os.listdir(dst)) > 0:
        print("skip", outname)
        continue
    print("extracting", zname, "->", dst, flush=True)
    t0 = time.time()
    with zipfile.ZipFile(zpath) as z:
        for i, e in enumerate(z.infolist()):
            if e.is_dir():
                continue
            name = e.filename
            if strip and name.startswith(strip + "/"):
                name = name[len(strip) + 1:]
            target = os.path.join(dst, name.replace("/", os.sep))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(e) as src, open(target, "wb") as fo:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    fo.write(b)
            if i % 5000 == 0:
                print(f"  {i} files {time.time()-t0:.1f}s", flush=True)
    print("done", outname, f"{time.time()-t0:.1f}s", flush=True)
