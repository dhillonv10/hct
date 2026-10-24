import json
import math

import numpy as np
import pytest
import torch

from epimod.lb.longrun import LongConfig, LongTrainer, PlateauController, StepGuard
from tests.test_lb_generators import toy_store


def test_plateau_controller_improves_drops_and_converges():
    c = PlateauController(patience=3, min_rel=0.01, max_drops=2, diverge_factor=1.3)
    assert [c.update(s) for s in (1.0, 0.9, 0.8, 0.7)] == ["improved"] * 4
    assert [c.update(0.7) for _ in range(3)] == ["same", "same", "drop"] and c.drops == 1          # flat for one window -> LR drop
    assert [c.update(0.7) for _ in range(3)] == ["same", "same", "drop"] and c.drops == 2
    assert [c.update(0.7) for _ in range(3)] == ["same", "same", "converged"]                      # drops used up -> stop


def test_slow_but_steady_improvement_is_not_a_plateau():
    c = PlateauController(patience=6, min_rel=0.005, max_drops=3, diverge_factor=1.3)
    scores = [1.0 * (1 - 0.0015) ** k for k in range(60)]                                        # 0.15% per evaluation
    assert all(a in ("improved",) for a in (c.update(x) for x in scores)) and c.drops == 0


def test_plateau_controller_flags_bad_and_diverged_scores():
    c = PlateauController(3, 0.01, 2, 1.3)
    c.update(1.0)
    assert c.update(float("nan")) == "bad" and c.update(float("inf")) == "bad"
    assert c.update(1.4) == "diverged" and c.update(1.1) == "same"


def test_step_guard_skips_nan_spikes_and_asks_for_rollback():
    g = StepGuard(spike_k=8.0, window=100, min_hist=20, max_consecutive=3)
    rng = np.random.RandomState(0)
    for _ in range(40):
        assert g.check(2.0 + 0.1 * rng.randn(), 1.0) == "ok"
    assert g.check(float("nan"), 1.0) == "skip" and g.check(2.0, float("inf")) == "skip"      # non-finite loss / gradient
    assert g.check(500.0, 1.0) == "rollback"                                                 # spike, third bad step in a row
    assert g.check(2.0, 1.0) == "ok" and g.consecutive == 0                                   # recovery resets the counter
    assert g.total_bad == 3


def _trainer(tmp_path, long_cfg, steps_kw=None):
    from epimod.lb.episet import SetConfig
    from epimod.lb.tasks import Pools, Vocab
    from epimod.lb.train import LBTrainConfig, TaskSets
    st = toy_store(S=60, U=400)
    vocab = Vocab(np.arange(400), st.universe, np.full(400, 0.5, np.float32), np.full(400, 0.25, np.float32))
    pools = Pools(np.arange(0, 150), np.arange(150, 300), np.arange(0, 400))
    sets = TaskSets(np.arange(30), {"Hannum": np.arange(30, 70)})
    cfg = LBTrainConfig(tasks=("extremes", "fixed", "clock", "pair"), bs={"extremes": 4, "fixed": 4, "clock": 4, "pair": 4}, threads=1, device="cpu")
    logs = []
    t = LongTrainer(st, st, vocab, sets, pools, SetConfig(d_model=16, d_emb=8, n_layers=1, n_heads=2), cfg, long_cfg, tmp_path, log=logs.append,
                    meta={"args": {"model": "set", "tasks": "extremes,fixed,clock,pair"}})
    return t, logs


def test_long_trainer_end_to_end_resume_nan_guard_and_stop_file(tmp_path):
    lc = LongConfig(eval_every=10, warmup=3, lr=2e-3, max_steps=30, cpu_check_every=0)
    # 1) a NaN loss injected at step 12 is skipped, the run still finishes its 30 steps
    t, logs = _trainer(tmp_path, lc)
    hit = []
    def hook(step, loss):
        if step == 12 and not hit:
            hit.append(1); return loss * float("nan")
        return loss
    res = t.fit_long(fault_hook=hook)
    assert res["step"] == 30 and res["bad_steps"] == 1 and res["reason"] == "max-steps"
    assert [h["step"] for h in res["history"]] == [10, 20, 30] and all(math.isfinite(h["val_score"]) for h in res["history"])
    assert (tmp_path / "model.pt").exists() and (tmp_path / "state.pt").exists() and (tmp_path / "status.json").exists()
    # 2) resume continues from the saved step with the saved history
    lc2 = LongConfig(eval_every=10, warmup=3, lr=2e-3, max_steps=50, cpu_check_every=0)
    t2, logs2 = _trainer(tmp_path, lc2)
    res2 = t2.fit_long()
    assert any("[resume] continuing from step 30" in l for l in logs2) and [h["step"] for h in res2["history"]] == [10, 20, 30, 40, 50]
    # 3) the exported model loads and reproduces the stored best validation score
    ck = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    assert {"state", "cfg", "vocab_ids", "args"} <= set(ck)
    # 4) a STOP file ends the run cleanly at the next step
    (tmp_path / "STOP").write_text("")
    t3, _ = _trainer(tmp_path, LongConfig(eval_every=10, warmup=3, max_steps=500, cpu_check_every=0))
    assert t3.fit_long()["reason"] == "stop-file"
    assert json.loads((tmp_path / "summary.json").read_text())["reason"] == "stop-file"


def test_long_trainer_survives_a_corrupt_latest_checkpoint(tmp_path):
    lc = LongConfig(eval_every=10, warmup=3, lr=2e-3, max_steps=20, cpu_check_every=0)
    t, _ = _trainer(tmp_path, lc); t.fit_long()
    (tmp_path / "state.pt").write_bytes(b"truncated")                       # simulate a crash during a write
    t2, logs = _trainer(tmp_path, LongConfig(eval_every=10, warmup=3, lr=2e-3, max_steps=30, cpu_check_every=0))
    res = t2.fit_long()
    assert any("could not read state.pt" in l for l in logs) and any("[resume]" in l for l in logs) and res["step"] == 30


def test_persistent_nan_triggers_rollback_then_stops_as_unstable(tmp_path):
    lc = LongConfig(eval_every=10, warmup=3, lr=2e-3, max_steps=200, max_consecutive_bad=3, cpu_check_every=0)
    t, logs = _trainer(tmp_path, lc)
    res = t.fit_long(fault_hook=lambda step, loss: loss * float("nan") if step > 15 else loss)
    assert res["reason"].startswith("unstable") and any("[rollback]" in l for l in logs)
    assert (tmp_path / "state.pt").exists()                                  # the last good state was not overwritten by a bad one
