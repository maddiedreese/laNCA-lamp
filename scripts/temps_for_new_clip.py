"""Starting temperatures for training on lamp-2026-11-04, from a run in the first clip's glass.

    uv run python scripts/temps_for_new_clip.py refit4_s0

Writes runs/physics/<name>_on_new. Cells only the new glass has take the nearest old cell's
temperature. A physics run can't supply these: the new clip's first frames are dark, so a run
started from them has almost no wax and stays cold.
"""

import argparse

import numpy as np
from scipy import ndimage

from lanca.extract import ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    name = ap.parse_args().run
    old = np.load(ROOT / "runs" / "physics" / name / "sim.npz")
    new_vessel = np.load(ROOT / "masks" / "lamp-2026-11-04_2fps.npz")["vessel"].astype(bool)
    nearest = ndimage.distance_transform_edt(
        ~old["vessel"].astype(bool), return_distances=False, return_indices=True
    )
    temps = old["temps"][:, nearest[0], nearest[1]] * new_vessel
    out = ROOT / "runs" / "physics" / f"{name}_on_new"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "sim.npz", temps=temps.astype(np.float32), vessel=new_vessel)


if __name__ == "__main__":
    main()
