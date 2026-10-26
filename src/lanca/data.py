"""Training data: windows of real wax masks, with a starting temperature for each.

The masks don't show temperature, so a window starts from a smooth temperature field: the
average temperature of the physics baseline over its later, cycling part. The model has to
work out the rest from how the wax moves.
"""

from dataclasses import dataclass

import numpy as np
import torch

from lanca.extract import ROOT

CLIP = "lamp-2026-10-25"


@dataclass
class Footage:
    masks: np.ndarray  # (T, H, W) uint8 wax probability
    fps: float
    vessel: np.ndarray  # (H, W) bool
    temp0: np.ndarray  # (H, W) float32 starting temperature

    @classmethod
    def load(cls, clip: str = CLIP, baseline: str = "default") -> "Footage":
        data = np.load(ROOT / "masks" / f"{clip}_2fps.npz")
        temps = np.load(ROOT / "runs" / "physics" / baseline / "sim.npz")["temps"].astype(
            np.float32
        )
        temp0 = temps[len(temps) // 2 :].mean(0)
        return cls(masks=data["masks"], fps=float(data["fps"]), vessel=data["vessel"], temp0=temp0)

    def frame_index(self, seconds: float) -> int:
        return round(seconds * self.fps)

    def windows(
        self, starts: np.ndarray, steps: int, device: str | torch.device = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Windows starting at frame indices `starts`, one frame per step.

        Returns the first frame's wax (B, H, W), starting temperature (B, H, W), and the
        following `steps` frames as targets (steps, B, H, W), all float32 in [0, 1].
        """
        idx = starts[:, None] + np.arange(steps + 1)[None]
        frames = torch.from_numpy(self.masks[idx].astype(np.float32) / 255).to(device)
        vessel = torch.from_numpy(self.vessel).to(device)
        frames = frames * vessel
        temp = torch.from_numpy(self.temp0).to(device)[None].expand(len(starts), -1, -1)
        return frames[:, 0], temp.contiguous(), frames[:, 1:].transpose(0, 1)
