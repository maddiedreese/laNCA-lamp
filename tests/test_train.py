from types import SimpleNamespace

import numpy as np
import torch
from test_physics import H, W, _vessel

from lanca import train
from lanca.data import Footage


def _footage(vessel: np.ndarray) -> Footage:
    masks = np.zeros((400, H, W), np.uint8)
    masks[:, H - 8 :] = 255
    return Footage(
        masks=masks * vessel,
        fps=0.1,  # keeps the last 15 minutes of footage to 90 frames
        vessel=vessel,
        temps=np.full((3, H, W), 0.5, np.float32) * vessel,
    )


def _trainer(monkeypatch, clips: list[str]) -> train.Trainer:
    full = _vessel().numpy().astype(bool)
    narrow = full.copy()
    narrow[:, W // 2 + 3 :] = False
    vessels = {"a": full, "b": narrow}
    monkeypatch.setattr(Footage, "load", lambda clip, baseline: _footage(vessels[clip]))
    torch.manual_seed(0)
    return train.Trainer(
        SimpleNamespace(
            device="cpu", clips=clips, temps=["t"] * len(clips), constants=None, init=None,
            stage="B", lr=1e-3, lr_physics=3e-3, iters=4, seed=0, curriculum=[2], warmup=2,
            batch=2, pool_batch=1, pool_size=4, pool_life=20, checkpoint=1, flow_grad=False,
            behavior_weight=1.0, eval_free=2,
        )
    )


def test_iterations_take_turns_and_lamps_stay_in_their_own_glass(monkeypatch):
    trainer = _trainer(monkeypatch, ["a", "b"])
    rows = [trainer.iteration(it) for it in range(1, 5)]
    assert [r["clip"] for r in rows] == ["b", "a", "b", "a"]
    for rec in trainer.recordings:
        assert rec.pool
        outside = ~rec.vessel.bool()
        assert all(state.wax[0][outside].abs().max() == 0 for state, _, _ in rec.pool)


def test_evaluation_averages_the_recordings(monkeypatch):
    trainer = _trainer(monkeypatch, ["a", "b"])
    both = trainer.evaluate()
    each = [trainer.evaluate_recording(rec) for rec in trainer.recordings]
    assert abs(both["frame"] - (each[0]["frame"] + each[1]["frame"]) / 2) < 1e-6


def test_mismatched_temps_are_refused(monkeypatch):
    monkeypatch.setattr(Footage, "load", lambda clip, baseline: None)
    args = SimpleNamespace(device="cpu", clips=["a", "b"], temps=["t"])
    try:
        train.Trainer(args)
    except ValueError:
        return
    raise AssertionError("expected a ValueError")
