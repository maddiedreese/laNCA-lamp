"""Training data: windows of real wax masks, with a starting temperature for each.

The masks don't show temperature, so a window starts from the temperature the physics
baseline had at the same point in the recording (snapshots once a minute). That gets the
broad state right: cold at the start, warming through the eruptions, hot while cycling. The
model has to work out the details from how the wax moves.
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
    temps: np.ndarray  # (minutes, H, W) float32 baseline temperature, one snapshot per minute

    @classmethod
    def load(cls, clip: str = CLIP, baseline: str = "default") -> "Footage":
        data = np.load(ROOT / "masks" / f"{clip}_2fps.npz")
        temps = np.load(ROOT / "runs" / "physics" / baseline / "sim.npz")["temps"]
        return cls(
            masks=data["masks"],
            fps=float(data["fps"]),
            vessel=data["vessel"],
            temps=temps.astype(np.float32),
        )

    def frame_index(self, seconds: float) -> int:
        return round(seconds * self.fps)

    def temp_at(self, frames: np.ndarray) -> np.ndarray:
        """Baseline temperature (B, H, W) at the given frame indices."""
        minute = np.minimum((frames / self.fps / 60).astype(int), len(self.temps) - 1)
        return self.temps[minute]

    def windows(
        self, starts: np.ndarray, steps: int, device: str | torch.device = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Windows starting at frame indices `starts`, one frame per step.

        Returns the first frame's wax (B, H, W), starting temperature (B, H, W), and the
        following `steps` frames as targets (steps, B, H, W), all float32 in [0, 1].
        """
        frames = self.frames(starts, steps, device)
        temp = torch.from_numpy(self.temp_at(starts)).to(device)
        return frames[0], temp, frames[1:]

    def frames(
        self, starts: np.ndarray, steps: int, device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        """Frames starts..starts+steps as (steps + 1, B, H, W) float32, masked to the vessel."""
        idx = starts[:, None] + np.arange(steps + 1)[None]
        frames = torch.from_numpy(self.masks[idx].astype(np.float32) / 255).to(device)
        return (frames * torch.from_numpy(self.vessel).to(device)).transpose(0, 1)
