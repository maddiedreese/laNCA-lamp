"""Decode a clip at a fixed rate, stabilize each frame against the lamp body, and crop the glass.

The camera in our footage drifts slowly over the recording, so every frame is aligned to the
first one using two rigid anchors on the lamp (the nose cone and the base). The alignment is a
similarity transform (shift, scale, small rotation) estimated from the two anchor positions.
"""

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOOTAGE = ROOT / "footage"


@dataclass
class ClipConfig:
    id: str
    path: Path
    decode_roi: tuple[int, int, int, int]
    anchors: dict[str, tuple[int, int, int, int]]
    glass_box: tuple[int, int, int, int]
    grid: tuple[int, int]

    @classmethod
    def load(cls, clip_id: str) -> "ClipConfig":
        manifest = json.loads((FOOTAGE / "manifest.json").read_text())
        clip = next(c for c in manifest["clips"] if c["id"] == clip_id)
        cfg = json.loads((FOOTAGE / "processing.json").read_text())[clip_id]
        return cls(
            id=clip_id,
            path=FOOTAGE / "clips" / clip["filename"],
            decode_roi=tuple(cfg["decode_roi"]),
            anchors={k: tuple(v) for k, v in cfg["anchors"].items()},
            glass_box=tuple(cfg["glass_box"]),
            grid=tuple(cfg["grid"]),
        )


def read_frames(
    path: Path,
    fps: float,
    roi: tuple[int, int, int, int],
    start: float = 0.0,
    duration: float | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (time_s, RGB frame cropped to roi) at a fixed rate using ffmpeg."""
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    # Software decoding: on Apple Silicon it is several times faster than VideoToolbox here,
    # because the hardware path copies every decoded frame back through the CPU anyway.
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += [
        "-i",
        str(path),
        "-vf",
        f"fps={fps},crop={w}:{h}:{x0}:{y0}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield start + i / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


def _subpixel_peak(r: np.ndarray, loc: tuple[int, int]) -> tuple[float, float]:
    """Refine an integer matchTemplate peak with a parabolic fit in x and y."""
    x, y = loc
    dx = dy = 0.0
    if 0 < x < r.shape[1] - 1:
        a, b, c = r[y, x - 1], r[y, x], r[y, x + 1]
        dx = 0.5 * (a - c) / (a - 2 * b + c + 1e-9)
    if 0 < y < r.shape[0] - 1:
        a, b, c = r[y - 1, x], r[y, x], r[y + 1, x]
        dy = 0.5 * (a - c) / (a - 2 * b + c + 1e-9)
    return x + dx, y + dy


class Stabilizer:
    """Aligns frames to a reference frame using template matching on rigid anchors.

    All coordinates are in the decode ROI's pixel space. Each anchor is searched for in a window
    around its last known position, so slow drift and the occasional small jump are both handled.
    """

    def __init__(
        self, reference: np.ndarray, anchors: dict[str, tuple[int, int, int, int]], search: int = 40
    ):
        gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
        self.search = search
        self.templates = {k: gray[y0:y1, x0:x1].copy() for k, (x0, y0, x1, y1) in anchors.items()}
        self.ref_pos = {
            k: np.array([x0, y0], np.float64) for k, (x0, y0, x1, y1) in anchors.items()
        }
        self.pos = dict(self.ref_pos)
        self.last_score: dict[str, float] = {}

    def locate(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        for k, t in self.templates.items():
            th, tw = t.shape
            px, py = self.pos[k].round().astype(int)
            x0, y0 = max(0, px - self.search), max(0, py - self.search)
            x1, y1 = min(w, px + tw + self.search), min(h, py + th + self.search)
            r = cv2.matchTemplate(gray[y0:y1, x0:x1], t, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(r)
            sx, sy = _subpixel_peak(r, loc)
            self.pos[k] = np.array([x0 + sx, y0 + sy])
            self.last_score[k] = float(score)
        return dict(self.pos)

    def transform(self, frame: np.ndarray) -> np.ndarray:
        """Return the 2x3 matrix mapping this frame's coordinates onto the reference frame."""
        cur = self.locate(frame)
        src = np.array([cur[k] for k in self.templates], np.float64)
        dst = np.array([self.ref_pos[k] for k in self.templates], np.float64)
        # Similarity from two point pairs, solved in closed form via complex numbers.
        s = complex(*src[1]) - complex(*src[0])
        d = complex(*dst[1]) - complex(*dst[0])
        a = d / s
        b = complex(*dst[0]) - a * complex(*src[0])
        return np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag]])


def stabilized_crops(
    cfg: ClipConfig, fps: float, start: float = 0.0, duration: float | None = None
) -> Iterator[tuple[float, np.ndarray, np.ndarray, dict[str, float]]]:
    """Yield (time_s, glass crop RGB at full resolution, 2x3 transform, anchor scores).

    The stabilizer's reference is always the clip's first frame, even when starting later,
    so crops from any segment line up with each other.
    """
    rx0, ry0, _, _ = cfg.decode_roi

    def to_roi(box):
        return box[0] - rx0, box[1] - ry0, box[2] - rx0, box[3] - ry0

    _, first = next(read_frames(cfg.path, 1.0, cfg.decode_roi, 0.0, 1.0))
    stab = Stabilizer(first, {k: to_roi(v) for k, v in cfg.anchors.items()})
    if start > 0:
        # Seed anchor positions near the start time so the search window is in the right place.
        stab.search = 160
        stab.transform(next(read_frames(cfg.path, 1.0, cfg.decode_roi, start, 1.0))[1])
        stab.search = 40

    gx0, gy0, gx1, gy1 = to_roi(cfg.glass_box)
    shift = np.array([[1, 0, -gx0], [0, 1, -gy0]], np.float64)
    for t, frame in read_frames(cfg.path, fps, cfg.decode_roi, start, duration):
        m = stab.transform(frame)
        # Compose: frame -> reference -> glass crop.
        m3 = np.vstack([m, [0, 0, 1]])
        crop_m = shift @ m3
        crop = cv2.warpAffine(
            frame,
            crop_m,
            (gx1 - gx0, gy1 - gy0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        yield t, crop, m, dict(stab.last_score)
