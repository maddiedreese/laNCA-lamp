"""Pick the compute device. Run `python -m lanca.device` to check the GPU is usable."""

import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    import time

    device = get_device()
    print(f"torch {torch.__version__}, device: {device}")

    # A grid the size of a lamp, run through a few hundred NCA-like conv steps.
    x = torch.rand(16, 16, 128, 64, device=device)
    conv = torch.nn.Conv2d(16, 16, 3, padding=1).to(device)
    with torch.no_grad():
        for _ in range(10):
            x = torch.tanh(conv(x))
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(500):
            x = torch.tanh(conv(x))
        if device.type == "mps":
            torch.mps.synchronize()
    print(f"500 steps on a 16x16x128x64 batch: {time.perf_counter() - start:.2f}s")
