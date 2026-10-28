import cv2
import numpy as np

from lanca import measure, segment
from lanca.extract import Stabilizer


def _scene(dx: float = 0.0, dy: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """A textured 300x200 image, shifted and scaled about the origin."""
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((300, 200)) * 255).astype(np.uint8), (0, 0), 2)
    m = np.array([[scale, 0, dx], [0, scale, dy]], np.float64)
    img = cv2.warpAffine(base, m, (200, 300), borderMode=cv2.BORDER_REFLECT)
    return np.repeat(img[..., None], 3, 2)


def test_stabilizer_undoes_shift_and_scale():
    anchors = {"top": (60, 40, 120, 80), "bottom": (60, 220, 120, 260)}
    stab = Stabilizer(_scene(), anchors, search=20)
    m = stab.transform(_scene(dx=7, dy=-11, scale=1.02))
    # The point the reference had at (90, 150) moved to 1.02 * (90, 150) + (7, -11).
    moved = np.array([1.02 * 90 + 7, 1.02 * 150 - 11, 1.0])
    # Anchors are matched without rescaling the templates, so allow about a pixel of error.
    np.testing.assert_allclose(m @ moved, [90, 150], atol=1.0)


def test_vessel_mask_fills_rows_and_ignores_grey():
    crop = np.full((40, 30, 3), 128, np.uint8)
    crop[5:35, 8:22] = (120, 60, 200)  # violet liquid
    crop[10:20, 10:14] = (255, 120, 20)  # orange wax inside it
    v = segment.vessel_mask(np.stack([crop] * 3), erode_px=0)
    assert v[20, 15] and v[15, 12]
    assert not v[2, 2] and not v[38, 28]


def test_wax_model_separates_wax_from_liquid():
    rng = np.random.default_rng(1)
    crops = np.zeros((20, 40, 30, 3), np.uint8)
    crops[:] = (120, 60, 200)
    for i, c in enumerate(crops):
        y = 5 + i
        c[y : y + 8, 10:20] = (255, 110, 30)
    crops = np.clip(crops + rng.integers(-5, 6, crops.shape), 0, 255).astype(np.uint8)
    vessel = np.ones((40, 30), bool)
    model = segment.WaxModel.fit(crops, vessel)
    p = model(crops[3])
    assert p[10, 15] > 0.9  # inside the blob (rows 8-15)
    assert p[30, 15] < 0.1  # liquid

    hot = crops[3].copy()
    hot[30:34, 12:18] = (255, 255, 255)  # clipped-white wax over the bulb
    assert model(hot)[32, 15] > 0.9


def test_to_grid_does_not_dilute_edges():
    soft = np.ones((20, 20), np.float32)
    vessel = np.zeros((20, 20), bool)
    vessel[:, 5:15] = True
    g = segment.to_grid(soft, vessel, (10, 10))
    assert g[5, 3] == 1.0  # cell covering columns 6-7, fully inside
    assert g[5, 0] == 0.0  # cell entirely outside the vessel


def _blob_frames(n: int = 10, speed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    masks = np.zeros((n, 64, 32), np.float32)
    for i in range(n):
        y = 40 - speed * i
        masks[i, y : y + 6, 12:18] = 1.0
        masks[i, 60:64, :] = 1.0  # pool
    return masks, np.ones((64, 32), bool)


def test_components_separate_pool_from_free_blob():
    masks, vessel = _blob_frames()
    _, blobs = measure.components(masks[0], vessel)
    assert sorted(b["pool"] for b in blobs) == [False, True]


def test_track_measures_rise_speed():
    masks, vessel = _blob_frames(speed=2)
    tr = measure.track(masks, vessel, fps=10.0)
    # 2 rows per frame at 10 fps over 63 rows of height.
    np.testing.assert_allclose(tr["velocity"], 2 * 10 / 63, rtol=1e-3)
    assert len(tr["merge_t"]) == 0 and len(tr["split_t"]) == 0


def test_track_measures_blob_share_of_all_wax():
    masks, vessel = _blob_frames()
    tr = measure.track(masks, vessel, fps=10.0)
    # A 6x6 blob next to a 4x32 pool.
    np.testing.assert_allclose(tr["blob_share"], 36 / (36 + 128), rtol=1e-6)
    # Twice the pool (more visible wax), same blob: its share halves-ish, its area doesn't.
    masks[:, 56:60, :] = 1.0
    tr2 = measure.track(masks, vessel, fps=10.0)
    np.testing.assert_allclose(tr2["blob_share"], 36 / (36 + 256), rtol=1e-6)
    np.testing.assert_allclose(tr2["blob_area"], tr["blob_area"])


def test_track_detects_merge():
    masks = np.zeros((2, 64, 32), np.float32)
    masks[0, 10:16, 4:10] = 1
    masks[0, 10:16, 14:20] = 1
    masks[1, 10:16, 4:20] = 1
    tr = measure.track(masks, np.ones((64, 32), bool), fps=1.0)
    assert len(tr["merge_t"]) == 1


def test_phases_find_first_blob_and_cycling():
    fps = 1.0
    free = np.zeros(3600, np.int32)
    free[600:620] = 1  # first blob breaks free at 10 min
    free[620:1500:2] = 1  # sporadic blobs, rolling mean below 1
    free[1500:] = 2  # regular cycling from 25 min
    ph = measure.phases({"free_blobs": free}, fps)
    assert ph["first_blob_s"] == 600
    assert 1200 <= ph["cycling_s"] <= 1500


def test_cycle_period_finds_sine_period():
    fps = 2.0
    t = np.arange(0, 3600, 1 / fps)
    period = measure.cycle_period(np.sin(2 * np.pi * t / 120), fps)
    assert abs(period - 120) < 1
