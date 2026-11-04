import os, sys, time, subprocess, argparse, json
RES = r"C:\wce\wce_experiment\results"
DATA = r"C:\wce\wce_experiment\data"
RUNS = r"C:\wce\wce_experiment\runs\controlled"
LOGS = os.path.join(RES, "train_logs")
TRAIN = r"C:\wce\WCE2024\Yolov5\yolov5\train.py"
WEIGHTS = r"C:\wce\WCE2024\Yolov5\yolov5s.pt"
HYP = r"C:\wce\wce_experiment\data\hyps\hyp.controlled.yaml"


def gpu_used():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode()
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="A,C,M1,M2,M3,P0,P2,E,P1,P1s,D")
    ap.add_argument("--seeds", default="42,2025,3407")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    conds = [c for c in args.conditions.split(",") if c]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    jobs = []
    for c in conds:
        for s in seeds:
            out = os.path.join(RUNS, f"{c}_s{s}")
            if os.path.exists(os.path.join(out, "weights", "best.pt")):
                continue
            jobs.append((c, s, out))
    print("jobs", len(jobs), flush=True)
    os.makedirs(LOGS, exist_ok=True)
    running = []
    idx = 0
    while idx < len(jobs) or running:
        while idx < len(jobs) and len(running) < args.parallel:
            if gpu_used() > 13500:
                break
            c, s, out = jobs[idx]
            log = os.path.join(LOGS, f"{c}_s{s}.log")
            cmd = [sys.executable, TRAIN, "--img", "576", "--batch", str(args.batch),
                   "--epochs", str(args.epochs), "--data", os.path.join(DATA, "yaml", c + ".yaml"),
                   "--weights", WEIGHTS, "--hyp", HYP, "--workers", str(args.workers),
                   "--project", RUNS, "--name", f"{c}_s{s}", "--seed", str(s),
                   "--patience", str(args.patience), "--exist-ok"]
            print("START", c, s, flush=True)
            if args.dry_run:
                print(" ".join(cmd)); idx += 1; continue
            f = open(log, "w")
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=r"C:\wce\wce_experiment")
            running.append((p, c, s, f, log))
            idx += 1
        if args.dry_run:
            continue
        time.sleep(20)
        still = []
        for p, c, s, f, log in running:
            if p.poll() is None:
                still.append((p, c, s, f, log))
            else:
                f.close()
                print("DONE", c, s, p.returncode, flush=True)
        running = still
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
