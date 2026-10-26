"""Turn stabilized glass crops into a vessel mask and soft wax masks.

Color model: the liquid is violet (blue-leaning, negative b* in Lab) and the wax is orange/red,
turning yellow-white near the bulb (positive b*). So b* separates them, with two corrections:

- Glare: the glass has fixed specular streaks. Frames are stabilized, so the streaks sit in the
  same pixels every frame; those pixels are filled in from their neighbors before thresholding.
- Bulb glow: liquid near the bulb is lit warm and its b* rises toward the wax range. Each pixel
  therefore gets its own threshold, measured from how that pixel looks when it holds liquid.
- Hot core: wax sitting right on the bulb glows so brightly that the camera clips it to white,
  and its b* falls back toward 0. Lit liquid never gets close to that bright (L* under ~160 of
  255 here), so near-saturated pixels count as wax.
"""

from dataclasses import dataclass

import cv2
import numpy as np


def lab(crop: np.ndarray) -> np.ndarray:
    """Float Lab with a* and b* centered on 0 (OpenCV 8-bit Lab offsets them by 128)."""
    x = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    x[..., 1:] -= 128
    return x


def vessel_mask(
    crops: np.ndarray, chroma_min: float = 12.0, coverage: float = 0.5, erode_px: int = 2
) -> np.ndarray:
    """Find the inside of the glass.

    Inside the glass every pixel is strongly colored (violet liquid or orange wax), while the
    wall, cap and base are grey. A pixel belongs to the vessel if it is colored in at least
    `coverage` of the sample frames. Each row is then filled between its outermost vessel pixels,
    and the result is eroded slightly to drop the glass edge, where refraction and glare live.
    """
    colored = np.stack([np.hypot(*lab(c)[..., 1:].transpose(2, 0, 1)) > chroma_min for c in crops])
    v = colored.mean(0) >= coverage
    v = cv2.morphologyEx(v.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(v)
    if n > 1:
        v = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    filled = np.zeros_like(v)
    for y in range(v.shape[0]):
        xs = np.nonzero(v[y])[0]
        if len(xs):
            filled[y, xs[0] : xs[-1] + 1] = 1
    return cv2.erode(filled, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)).astype(bool)


def glare_mask(crops: np.ndarray, vessel: np.ndarray, excess: float = 12.0) -> np.ndarray:
    """Pixels that are consistently much brighter than their surroundings: fixed reflections."""
    med = np.median(np.stack([lab(c)[..., 0] for c in crops]), 0)
    local = cv2.medianBlur(np.clip(med, 0, 255).astype(np.uint8), 21).astype(np.float32)
    glare = ((med - local) > excess) & vessel
    return cv2.dilate(glare.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def otsu(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    return float(mids[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


@dataclass
class WaxModel:
    """Soft wax probability, with glare filled in first:

    max(sigmoid((b* - threshold[y, x]) / softness), sigmoid((L* - hot_l) / hot_softness))
    """

    threshold: np.ndarray
    softness: float
    glare: np.ndarray
    params: dict
    hot_l: float = 215.0
    hot_softness: float = 8.0

    @classmethod
    def fit(
        cls, crops: np.ndarray, vessel: np.ndarray, liquid_pct: float = 10.0, glow_cap: float = 25.0
    ) -> "WaxModel":
        glare = glare_mask(crops, vessel)
        b = np.stack([_fill_glare(lab(c)[..., 2], glare) for c in crops])
        bv = b[:, vessel]
        t_otsu = otsu(bv)
        # Otsu returns the first threshold of a flat optimum, which can sit right on the edge of
        # the liquid class; recenter between the two class means (one isodata step).
        lo, hi = bv[bv < t_otsu].mean(), bv[bv >= t_otsu].mean()
        t = float((lo + hi) / 2)

        # How each pixel looks as liquid: a low percentile over time (it holds liquid most of the
        # time). Capped so pixels that are nearly always wax, like the pool, keep a sane threshold.
        liquid = np.percentile(b, liquid_pct, axis=0)
        ref = float(np.median(liquid[vessel]))
        liquid = cv2.GaussianBlur(np.clip(liquid, ref, ref + glow_cap), (0, 0), 2)
        threshold = liquid + (t - ref)
        params = {
            "otsu_b": t_otsu,
            "threshold_b": t,
            "liquid_b_ref": ref,
            "liquid_pct": liquid_pct,
            "glow_cap": glow_cap,
            "class_means_b": [float(lo), float(hi)],
            "glare_px": int(glare.sum()),
        }
        return cls(
            threshold=threshold.astype(np.float32),
            softness=float(max(1.0, (hi - lo) / 8)),
            glare=glare,
            params=params,
        )

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        x = lab(crop)
        b = _fill_glare(x[..., 2], self.glare)
        light = _fill_glare(x[..., 0], self.glare)
        colored = 1.0 / (1.0 + np.exp(-(b - self.threshold) / self.softness))
        hot = 1.0 / (1.0 + np.exp(-(light - self.hot_l) / self.hot_softness))
        return np.maximum(colored, hot)


def _fill_glare(channel: np.ndarray, glare: np.ndarray) -> np.ndarray:
    if not glare.any():
        return channel
    return cv2.inpaint(channel.astype(np.float32), glare.astype(np.uint8), 3, cv2.INPAINT_TELEA)


def to_grid(soft: np.ndarray, vessel: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Average a soft mask down to the simulation grid (H, W).

    Only pixels inside the vessel count, so cells on the glass edge are not diluted by the
    pixels outside it.
    """
    h, w = grid
    v = vessel.astype(np.float32)
    num = cv2.resize(soft.astype(np.float32) * v, (w, h), interpolation=cv2.INTER_AREA)
    den = cv2.resize(v, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(num / np.maximum(den, 1e-6), 0, 1) * (den > 0)


def grid_vessel(vessel: np.ndarray, grid: tuple[int, int], min_cover: float = 0.5) -> np.ndarray:
    h, w = grid
    return cv2.resize(vessel.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= min_cover
