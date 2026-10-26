import json
from pathlib import Path

import cv2
import torch

from lanca.device import get_device

ROOT = Path(__file__).resolve().parents[1]


def test_device_runs_a_conv():
    device = get_device()
    x = torch.rand(1, 16, 128, 64, device=device)
    y = torch.nn.Conv2d(16, 16, 3, padding=1).to(device)(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_opencv_imports():
    assert cv2.__version__


def test_manifest_is_valid_json():
    manifest = json.loads((ROOT / "footage" / "manifest.json").read_text())
    assert isinstance(manifest["clips"], list)
