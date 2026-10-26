"""Behavior statistics measured from wax masks.

These are the targets the trained lamp has to reproduce (Phase 5) and the benchmarks it is
judged against (Phase 6). Lengths are in lamp heights (1.0 = the full glass), times in seconds.
"""

import cv2
import numpy as np

MIN_BLOB_PX = 4  # components smaller than this are treated as segmentation noise
POOL_ROWS = 3  # a component touching the bottom this many rows is the pool, not a free blob


def components(mask: np.ndarray, vessel: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Label wax regions in one soft mask. Returns the label image and per-blob info."""
    h = mask.shape[0]
    binary = ((mask >= 0.5) & vessel).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    vessel_bottom = np.nonzero(vessel.any(1))[0].max()
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_PX:
            labels[labels == i] = 0
            continue
        bottom = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] - 1
        blobs.append(
            {
                "label": i,
                "area": int(area),
                "cx": float(cents[i, 0]),
                "height": float((h - 1 - cents[i, 1]) / (h - 1)),
                "pool": bool(bottom >= vessel_bottom - POOL_ROWS + 1),
            }
        )
    return labels, blobs


def per_frame(masks: np.ndarray, vessel: np.ndarray) -> dict[str, np.ndarray]:
    """Cheap per-frame series: wax area, centroid height, wax in the top half, free-blob count."""
    h = masks.shape[1]
    v = vessel.astype(np.float32)
    rows = np.linspace(1, 0, h, dtype=np.float32)[None, :, None]
    m = masks.astype(np.float32) * v
    area = m.sum((1, 2))
    vessel_area = v.sum()
    top = m[:, : h // 2].sum((1, 2)) / vessel_area
    height = (m * rows).sum((1, 2)) / np.maximum(area, 1e-6)
    free = np.zeros(len(masks), np.int32)
    for i, mk in enumerate(masks):
        _, blobs = components(mk, vessel)
        free[i] = sum(not b["pool"] for b in blobs)
    return {"area": area / vessel_area, "top_half": top, "height": height, "free_blobs": free}


def track(masks: np.ndarray, vessel: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Follow free blobs between consecutive frames by overlap.

    Returns vertical velocities of free blobs (lamp heights / s), blob areas (fraction of the
    vessel), and merge/split event times.
    """
    vessel_area = float(vessel.sum())
    velocities, areas, merges, splits = [], [], [], []
    prev_labels, prev_blobs = components(masks[0], vessel)
    for i in range(1, len(masks)):
        labels, blobs = components(masks[i], vessel)
        by_label = {b["label"]: b for b in blobs}
        prev_by_label = {b["label"]: b for b in prev_blobs}
        pairs = np.stack([prev_labels.ravel(), labels.ravel()], 1)
        pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
        links = np.unique(pairs, axis=0) if len(pairs) else np.zeros((0, 2), int)
        links = [(a, b) for a, b in links if a in prev_by_label and b in by_label]

        sources: dict[int, list[int]] = {}
        targets: dict[int, list[int]] = {}
        for a, b in links:
            sources.setdefault(b, []).append(a)
            targets.setdefault(a, []).append(b)
        t = i / fps
        merges += [t] * sum(len(s) > 1 for s in sources.values())
        splits += [t] * sum(len(s) > 1 for s in targets.values())
        for b, srcs in sources.items():
            if len(srcs) == 1 and len(targets[srcs[0]]) == 1:
                cur, old = by_label[b], prev_by_label[srcs[0]]
                if not cur["pool"] and not old["pool"]:
                    velocities.append((cur["height"] - old["height"]) * fps)
                    areas.append(cur["area"] / vessel_area)
        prev_labels, prev_blobs = labels, blobs
    return {
        "velocity": np.array(velocities, np.float32),
        "blob_area": np.array(areas, np.float32),
        "merge_t": np.array(merges, np.float32),
        "split_t": np.array(splits, np.float32),
    }


def cycle_period(
    series: np.ndarray, fps: float, min_s: float = 10.0, max_s: float = 900.0
) -> float | None:
    """Dominant period of a series from its autocorrelation, or None if there is no clear peak."""
    x = series - series.mean()
    if not x.any():
        return None
    ac = np.correlate(x, x, "full")[len(x) - 1 :]
    ac /= ac[0]
    lo, hi = int(min_s * fps), min(int(max_s * fps), len(ac) - 1)
    seg = ac[lo:hi]
    peaks = np.nonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] > 0.1))[0]
    if not len(peaks):
        return None
    return float((lo + 1 + peaks[np.argmax(seg[1:-1][peaks])]) / fps)


def phases(
    series: dict[str, np.ndarray], fps: float, hold_s: float = 10.0
) -> dict[str, float | None]:
    """Find warm-up milestones from the per-frame series.

    first_blob: first time a free blob exists continuously for hold_s.
    cycling: start of regular cycling, the point after which the 5-minute rolling mean of the
    free-blob count never drops below 1. Earlier, the wax rises as plumes and columns that stay
    attached to the pool.

    There is deliberately no "steady state": in our recording the wax keeps drifting upward
    through the final hour, so the lamp is still maturing when the footage ends.
    """
    hold = max(1, int(hold_s * fps))
    free = series["free_blobs"] > 0
    run = np.convolve(free.astype(int), np.ones(hold, int), "valid")
    idx = np.nonzero(run == hold)[0]
    first_blob = float(idx[0] / fps) if len(idx) else None

    win = int(300 * fps)
    cycling = None
    if len(free) > win:
        roll = np.convolve(series["free_blobs"].astype(np.float32), np.ones(win) / win, "valid")
        bad = np.nonzero(roll < 1.0)[0]
        start = bad[-1] + 1 if len(bad) else 0
        if start < len(roll):
            cycling = float(start / fps)
    return {"first_blob_s": first_blob, "cycling_s": cycling}


def _quantiles(x: np.ndarray) -> dict[str, float] | None:
    if not len(x):
        return None
    return {k: float(v) for k, v in zip(["p10", "p50", "p90"], np.percentile(x, [10, 50, 90]))}


def _motion(masks: np.ndarray, vessel: np.ndarray, fps: float, track_fps: float) -> dict:
    """Blob motion and topology stats, tracked on time-averaged masks at a lower rate.

    Averaging over each tracking interval stops threshold flicker at blob boundaries from
    registering as merge/split pairs. Merges and splits are contacts in the 2D view: many are
    blobs passing in front of each other, which show up as a merge quickly undone by a split.
    brief_merge_fraction is the share of merges followed by a split within 2 s.
    """
    step = max(1, round(fps / track_fps))
    n = len(masks) // step * step
    smooth = masks[:n].astype(np.float32).reshape(-1, step, *masks.shape[1:]).mean(1)
    tr = track(smooth, vessel, fps / step)
    minutes = n / fps / 60
    rising = tr["velocity"][tr["velocity"] > 0.002]
    sinking = -tr["velocity"][tr["velocity"] < -0.002]
    mt, st = tr["merge_t"], tr["split_t"]
    brief = sum(bool(np.any((st > t) & (st <= t + 2.0))) for t in mt)
    return {
        "rise_speed": _quantiles(rising),
        "sink_speed": _quantiles(sinking),
        "blob_area": _quantiles(tr["blob_area"]),
        "merges_per_min": len(tr["merge_t"]) / minutes if minutes else None,
        "splits_per_min": len(tr["split_t"]) / minutes if minutes else None,
        "brief_merge_fraction": brief / len(mt) if len(mt) else None,
    }


def summarize(
    masks: np.ndarray,
    vessel: np.ndarray,
    fps: float,
    track_fps: float = 2.0,
    window_s: float = 600.0,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Full statistics for one clip.

    "cycling" holds distributions over everything from the start of cycling to the end.
    "windows" breaks the same period into fixed windows, to show the slow drift.
    """
    series = per_frame(masks, vessel)
    ph = phases(series, fps)
    s = int((ph["cycling_s"] or 0.0) * fps)

    def block(a: int, b: int) -> dict:
        m = masks[a:b]
        profile = (m.astype(np.float32) * vessel).mean(0).sum(1) / np.maximum(vessel.sum(1), 1)
        return {
            "wax_area": _quantiles(series["area"][a:b]),
            "top_half": _quantiles(series["top_half"][a:b]),
            "height": _quantiles(series["height"][a:b]),
            "free_blobs": _quantiles(series["free_blobs"][a:b]),
            "cycle_period_s": cycle_period(series["top_half"][a:b], fps),
            **_motion(m, vessel, fps, track_fps),
            "vertical_profile": profile.round(4).tolist(),
        }

    win = int(window_s * fps)
    windows = []
    for a in range(s, len(masks) - win // 2, win):
        b = min(len(masks), a + win)
        w = block(a, b)
        w.pop("vertical_profile")
        windows.append({"start_s": a / fps, "end_s": b / fps, **w})
    return {
        "fps": fps,
        "duration_s": len(masks) / fps,
        "phases": ph,
        "cycling": block(s, len(masks)),
        "windows": windows,
    }, series
