"""Long, unattended training of EpiTrajectory-L on a Mac (Apple-silicon GPU / MPS only).

    python scripts/train_mps.py --name run1 --preset base            # first: hours, to see it work
    python scripts/train_mps.py --name run2 --preset large           # then: as long as validation keeps improving
    touch runs/run1/STOP                                             # ends a run cleanly at the next step
    cat runs/run1/status.json                                        # progress at any time

What happens: (1) a pre-flight on the real data - the GPU must reproduce the CPU's loss and gradients for every task, the
loss on one fixed batch must fall below 60%, the mini-run must not diverge, skip an injected NaN, resume from its checkpoint and export a model that scores the same on the CPU;
(2) the training worker runs under a supervisor that restarts it from its last checkpoint after a crash (and gives up if it
keeps crashing without progress); (3) training stops by itself when validation stops improving (LR halved up to 3 times).
`caffeinate` keeps the Mac awake for the duration.  Output: runs/<name>/{model.pt (best), status.json, summary.json, train.log}."""
import argparse, hashlib, json, os, subprocess, sys, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")            # ops without a Metal kernel fall back to the CPU instead of failing (set before torch is imported)
from pathlib import Path

DEBUG_DEVICE = os.environ.get("EPIMOD_DEBUG_DEVICE")                 # developer override (CPU testing); a normal run is MPS-only

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--out", default="runs")
ap.add_argument("--preset", default="base", choices=["base", "large"]); ap.add_argument("--model", default="hier", choices=["hier", "set"])
ap.add_argument("--tasks", default="extremes,fixed,clock,pair,mask"); ap.add_argument("--bs-mult", type=float, default=4.0)
ap.add_argument("--lr", type=float, default=0.0, help="0 = preset default"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-unlabeled", dest="unlabeled", action="store_false", help="do not use the label-free GEO series for masked reconstruction")
ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--min-rel", type=float, default=0.005, help="required relative improvement of the best score per patience window")
ap.add_argument("--max-lr-drops", type=int, default=3); ap.add_argument("--max-hours", type=float, default=0.0)
ap.add_argument("--max-restarts", type=int, default=30); ap.add_argument("--skip-preflight", action="store_true")
ap.add_argument("--preflight-steps", type=int, default=240); ap.add_argument("--preflight-only", action="store_true")
ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
a = ap.parse_args()
run_dir = Path(a.out) / a.name; run_dir.mkdir(parents=True, exist_ok=True)
LOG = run_dir / "train.log"


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def device():
    import torch
    if DEBUG_DEVICE:
        return torch.device(DEBUG_DEVICE)
    if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        sys.exit("This script needs the Apple-silicon GPU (MPS): torch.backends.mps is not available here.\n"
                 "Use a native arm64 Python (python3 -c 'import platform; print(platform.machine())' must print arm64) and torch >= 2.2.")
    return torch.device("mps")


def make_trainer_factory(inputs):
    from epimod.lb.longrun import LongTrainer
    meta = {"args": {"model": a.model, "tasks": a.tasks, "name": a.name, "preset": a.preset}}
    def make(rd, lc):
        return LongTrainer(inputs["tr"], inputs["va"], inputs["vocab"], inputs["sets"], inputs["pools"], inputs["mcfg"], inputs["cfg"], lc, rd, log=log, meta=meta)
    return make


def setup():
    import torch
    from epimod.lb.longrun import build_inputs
    dev = device(); torch.manual_seed(a.seed)
    log(f"loading data (unlabeled={a.unlabeled}) ...")
    inputs = build_inputs(a, dev.type)
    log(f"train {inputs['tr'].n} samples | val {inputs['va'].n} samples | vocab {len(inputs['vocab'])} | preset {a.preset} | device {dev}")
    return dev, inputs


def worker():
    from epimod.lb.longrun import LongConfig
    dev, inputs = setup()
    lc = LongConfig(eval_every=a.eval_every, lr=inputs["lr"], patience=a.patience, min_rel=a.min_rel, max_lr_drops=a.max_lr_drops, max_hours=a.max_hours)
    T = make_trainer_factory(inputs)(run_dir, lc)
    n_params = sum(p.numel() for p in T.model.parameters())
    log(f"parameters {n_params:,}; lr {lc.lr}; batch sizes {inputs['cfg'].bs}")
    res = T.fit_long()
    sys.exit(0 if res["reason"] in ("stop-file", "max-hours", "max-steps") or res["reason"].startswith("converged") else 3)


def preflight_only():
    from epimod.lb import preflight
    dev, inputs = setup()
    ok = preflight.run(make_trainer_factory(inputs), run_dir / "preflight", inputs["lr"], dev, steps=a.preflight_steps, log=log)
    sys.exit(0 if ok else 2)


def stamp():
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    return hashlib.md5(json.dumps([git, a.preset, a.model, a.tasks, a.bs_mult, a.unlabeled, a.seed]).encode()).hexdigest()


def supervisor():
    device()                                                                   # fail fast when there is no MPS
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])         # no idle sleep while this process lives
    me, passthrough = [sys.executable, str(Path(__file__).resolve())], [x for x in sys.argv[1:]]
    marker = run_dir / "preflight.ok"
    if not a.skip_preflight and not (marker.exists() and marker.read_text() == stamp()):
        log("pre-flight checks ...")
        if subprocess.call(me + [x for x in passthrough if x not in ("--skip-preflight",)] + ["--preflight-only"]) != 0:
            sys.exit("PRE-FLIGHT FAILED - nothing was started. See the [FAIL] lines above (also in " + str(LOG) + ").")
        marker.write_text(stamp())
    if a.preflight_only:
        return
    restarts, stuck, last = 0, 0, -1
    while True:
        log(f"starting worker (restart {restarts})")
        rc = subprocess.call(me + [x for x in passthrough if x != "--preflight-only"] + ["--worker"])
        if rc == 0:
            log("finished: " + (run_dir / "summary.json").read_text().replace("\n", " ")); break
        if rc == 3:
            sys.exit("Stopped for a reason that needs a human (unstable training or a device mismatch): see " + str(run_dir / "summary.json") + " and " + str(LOG))
        restarts += 1
        try:
            step = json.loads((run_dir / "status.json").read_text())["step"]
        except Exception:
            step = -1
        stuck = stuck + 1 if step <= last else 0; last = step
        log(f"worker exited with code {rc}; last checkpointed step {step}; restart {restarts}/{a.max_restarts}")
        if stuck >= 3 or restarts > a.max_restarts:
            sys.exit("Giving up: the worker keeps crashing without making progress. See " + str(LOG))
        time.sleep(min(600, 20 * restarts))


if __name__ == "__main__":
    if a.preflight_only:
        preflight_only()
    elif a.worker:
        worker()
    else:
        supervisor()
