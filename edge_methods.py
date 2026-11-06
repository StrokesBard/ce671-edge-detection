"""
edge_methods.py
===============
Image-processing logic for the CE671 term project
"Edge detection for road and river identification in satellite imagery".

Every function here is *pure*: NumPy array + plain parameters in,
array / float / tuple out. Nothing here imports Streamlit, so the same
functions can be called from a plain script or notebook to make report
figures.

Conventions
- Grayscale images are 2-D uint8 arrays.
- Edge maps and masks are 2-D uint8 arrays with values {0, 255}
  (255 = edge / feature pixel).
"""

import cv2
import numpy as np

WORK_SIZE = 512                 # longer image side used for all processing
SMOOTHING_METHODS = ("none", "average", "median", "gaussian")
# Chosen on the tune split by `python evaluate.py roads`
DEFAULT_HOUGH = {"threshold": 50, "min_line_length": 20, "max_line_gap": 20}


# ---------------------------------------------------------------------------
# Shared preprocessing (bundled samples and uploads both go through this)
# ---------------------------------------------------------------------------

def prepare_image(img, max_side=WORK_SIZE):
    """Resize so the longer side is at most `max_side` px, then make grayscale.

    INTER_AREA is the correct interpolation for downsampling (it averages
    the source pixels instead of skipping them). Small images are never
    upscaled, since that would only add interpolation blur. A single-band
    image (e.g. NDWI) is resized the same way and passed through as its own
    grayscale. Returns (img_resized, gray_resized).
    """
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return img, gray


def sentinel2_to_bgr(bands, low_pct=2, high_pct=98):
    """True-colour 8-bit BGR image from a Sentinel-2 band stack (H, W, 12).

    Assumes the SWED band order B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9,
    B11, B12, so blue / green / red (B2 / B3 / B4) are indices 1 / 2 / 3.
    Reflectance values (x 10 000) are stretched linearly between the 2nd
    and 98th percentile of the three bands together, which keeps the
    colour balance.
    """
    rgb = bands[..., [1, 2, 3]].astype(np.float32)   # B2, B3, B4 = B, G, R
    lo, hi = np.percentile(rgb, [low_pct, high_pct])
    bgr = np.clip((rgb - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    # Band selection leaves a non-standard memory layout, which OpenCV's
    # in-place drawing functions reject; make it a plain C-ordered array.
    return np.ascontiguousarray(bgr.astype(np.uint8))


def ndwi(bands):
    """Normalized Difference Water Index (McFeeters, 1996), in [-1, 1].

        NDWI = (Green - NIR) / (Green + NIR)
        Green = B3 (index 2), NIR = B8 (index 7) in the SWED band order

    Water reflects some green light but absorbs almost all near-infrared,
    so NDWI > 0 over water. Vegetation reflects NIR strongly (NDWI well
    below 0), and soil and built-up areas fall in between. The water-land
    boundary therefore becomes the strongest edge in the image, while the
    differences between fields, which all reflect NIR, are squeezed
    together. That is exactly what true-colour grayscale fails to do:
    there, turbid water and green fields have similar brightness.
    """
    green = bands[..., 2].astype(np.float32)
    nir = bands[..., 7].astype(np.float32)
    return (green - nir) / np.maximum(green + nir, 1e-6)


def ndwi_to_gray(index):
    """NDWI as an 8-bit image, water bright: -1 -> 0, 0 -> 128, +1 -> 255.

    A fixed mapping rather than a per-image stretch, so the same gray level
    means the same NDWI in every patch and Canny thresholds carry over.
    """
    return np.clip((index + 1) * 127.5, 0, 255).astype(np.uint8)


def prepare_mask(mask, shape):
    """Bring a ground-truth mask to `shape` (rows, cols) and binarise it.

    Nearest-neighbour resizing keeps the mask binary; the > 127 threshold
    cleans up the non-0/255 values that JPEG-ish masks often contain.
    White (255) = road / river.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sobel: gradients, edge strength (magnitude) and edge direction
# ---------------------------------------------------------------------------

def sobel_gradients(img, ksize=3):
    """Raw Sobel derivatives (Gx, Gy) as float32 arrays.

    Gx = dI/dx responds to *vertical* edges (intensity changes along x),
    Gy = dI/dy responds to *horizontal* edges (intensity changes along y).
    A larger ksize smooths more along the edge, and its responses are
    larger in absolute value, so compare maps of the same ksize only.
    """
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    return gx, gy


def sobel_magnitude_direction(img, ksize=3):
    """Edge strength and edge direction, using the lecture formulas.

        magnitude   M(x, y) = sqrt(Gx^2 + Gy^2)
        direction   theta(x, y) = arctan2(Gy, Gx)   (radians, -pi..pi)

    Why arctan2(Gy, Gx) instead of arctan(Gy / Gx):
    - Gy / Gx divides by zero wherever Gx = 0 (every purely horizontal edge).
    - arctan only returns -90..+90 deg. Flipping both signs gives the same
      ratio, (-Gy)/(-Gx) = Gy/Gx, so it cannot tell a dark-to-bright edge
      from a bright-to-dark one. arctan2 uses the sign of each component
      and covers the full circle.

    theta is the *gradient* direction: it points across the edge, from
    dark to bright. The edge itself runs perpendicular to it (theta +/- 90
    deg). Image rows grow downwards, so on screen theta is measured
    clockwise from the +x axis.
    """
    gx, gy = sobel_gradients(img, ksize)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)
    return magnitude, direction


def sobel_edge(img, ksize=3):
    """Binary Sobel edge map: threshold the gradient magnitude.

    The threshold is picked automatically with Otsu's method, so Sobel has
    no threshold to tune. Sobel has no thinning step (unlike Canny's
    non-maximum suppression), so edges come out several pixels thick, and
    every strongly textured patch (roofs, tree crowns, field rows) shows up
    along with the roads and rivers.
    """
    magnitude, _ = sobel_magnitude_direction(img, ksize)
    # Scale by the 99.5th percentile rather than the max, so a few extreme
    # pixels don't squash the rest of the histogram before Otsu.
    scale = max(float(np.percentile(magnitude, 99.5)), 1e-6)
    mag8 = np.clip(magnitude / scale * 255, 0, 255).astype(np.uint8)
    _, edges = cv2.threshold(mag8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges


# ---------------------------------------------------------------------------
# Canny and the two feature-specific pipelines
# ---------------------------------------------------------------------------

def canny_edge(img, low_thresh, high_thresh):
    """Canny edge detector.

    OpenCV's Canny does: Sobel gradient -> non-maximum suppression (thins
    edges to 1 px along the gradient direction) -> double threshold with
    hysteresis:
        magnitude >= high          strong edge, always kept
        low <= magnitude < high    weak edge, kept only if connected to a
                                   strong edge
        magnitude < low            discarded
    L2gradient=True makes it use the exact sqrt(Gx^2 + Gy^2) magnitude
    instead of the |Gx| + |Gy| shortcut.

    Textbook Canny starts with Gaussian smoothing. cv2.Canny does not, so
    the caller smooths first with smooth_image(). That keeps the
    smoothing step swappable for the noise-robustness comparison.
    """
    low, high = sorted((low_thresh, high_thresh))
    return cv2.Canny(img, low, high, L2gradient=True)


def canny_hough_roads(img, low_thresh, high_thresh, hough_params=None):
    """Roads: Canny followed by a probabilistic Hough line transform.

    Why it helps: at this resolution roads are long, locally straight
    features. Most of the clutter Canny also finds (building outlines, tree
    crowns, field texture) consists of short or curved fragments. The Hough
    step keeps only edge pixels that line up into long straight segments.

    Steps
    1. Canny edge map (binary, 1 px wide).
    2. Hough voting. A line is written in polar form
           rho = x * cos(theta) + y * sin(theta)
       Every edge pixel (x, y) votes for all (rho, theta) pairs of lines
       that pass through it. Pixels that lie on one straight line all vote
       for the same (rho, theta) cell, so that cell's count grows with the
       length of the line.
    3. Cells with at least `threshold` votes are accepted as lines. The
       *probabilistic* variant (HoughLinesP) returns finite segments with
       end points instead of infinite lines, and applies two more rules:
         - min_line_length: shorter segments are dropped (removes clutter)
         - max_line_gap: collinear pieces separated by at most this many
           px are joined into one segment (bridges road edges broken by
           trees, cars or shadows)
    4. Draw the surviving segments on an empty canvas. The output therefore
       contains only straight-line structure.
    """
    p = {**DEFAULT_HOUGH, **(hough_params or {})}
    edges = canny_edge(img, low_thresh, high_thresh)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,                    # accumulator resolution: 1 px in rho
        theta=np.pi / 180,        # ... and 1 degree in theta
        threshold=int(p["threshold"]),
        minLineLength=int(p["min_line_length"]),
        maxLineGap=int(p["max_line_gap"]),
    )

    out = np.zeros_like(edges)
    if lines is not None:
        # OpenCV 4 returns shape (N, 1, 4), OpenCV 5 returns (N, 4)
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            # 1 px thick, same as Canny output, so the scores stay comparable
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
    return out


def canny_morph_rivers(img, low_thresh, high_thresh, kernel_size=7,
                       min_length=None):
    """Rivers: Canny followed by morphological closing.

    Why it helps: where a bank has low contrast (turbid water next to wet
    soil or riparian vegetation) the gradient drops below Canny's
    thresholds in places, so the bank comes out as a dashed line of short
    fragments.

    Steps
    1. Canny edge map.
    2. Morphological CLOSING = dilation, then erosion, with the same
       structuring element (an ellipse of kernel_size x kernel_size, so no
       direction is preferred; banks curve):
         - dilation grows every edge pixel into a disc. Fragments separated
           by a gap narrower than about the kernel size now touch and merge.
         - erosion shrinks everything back by the same disc. Edges return to
           roughly their original width and position, but the bridges that
           filled the gaps survive, because they are now inside the merged
           shape.
       Net effect: gaps and holes smaller than the kernel are filled
       without shifting the edges.
    3. Short-fragment removal. After closing, a bank is one long connected
       curve, while texture edges stay small isolated blobs. Connected
       components whose bounding box is shorter than `min_length` px
       (default: 10% of the longer image side) are deleted.
    """
    edges = canny_edge(img, low_thresh, high_thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    if min_length is None:
        min_length = 0.1 * max(img.shape[:2])
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed,
                                                           connectivity=8)
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH],
                        stats[:, cv2.CC_STAT_HEIGHT])
    keep = extent >= min_length
    keep[0] = False                               # label 0 is the background
    return np.where(keep[labels], 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Noise and smoothing (median vs Gaussian on salt-and-pepper noise)
# ---------------------------------------------------------------------------

def add_salt_pepper_noise(img, amount=0.05, seed=0):
    """Corrupt a fraction `amount` of pixels: half to 0 (pepper), half to
    255 (salt).

    Salt-and-pepper (impulse) noise models dropped or saturated detector
    pixels and transmission bit errors. The fixed seed gives the same noise
    pattern on every call, so moving other sliders doesn't reshuffle the
    noise and the scores stay comparable.
    """
    rng = np.random.default_rng(seed)
    r = rng.random(img.shape[:2])
    noisy = img.copy()
    noisy[r < amount / 2] = 0
    noisy[(r >= amount / 2) & (r < amount)] = 255
    return noisy


def smooth_image(img, method="gaussian", kernel_size=5):
    """Smooth with one of SMOOTHING_METHODS using a k x k window.

    average   box filter, every pixel = plain mean of the window (linear)
    gaussian  weighted mean, weights fall off with distance; sigma is
              derived from k by OpenCV (linear)
    median    every pixel = median of the window (non-linear, rank order)

    The lecture claim, "median beats Gaussian on salt-and-pepper noise":
    - An impulse (0 or 255) is an extreme outlier. A *linear* filter
      multiplies it by a weight and adds it into every neighbour, so the
      impulse is not removed. It is smeared into a small bright or dark
      blob, which still has a strong gradient around its rim, so the edge
      detector fires on it.
    - The *median* sorts the window and takes the middle value. While fewer
      than half of the window's pixels are corrupted, the impulses sit at
      the two ends of the sorted list and are never chosen, so they are
      removed completely. The output is always an actual neighbouring
      value, so a true step edge (road or river bank) stays sharp instead
      of being blurred the way a Gaussian blurs it.
    """
    k = int(kernel_size) | 1                      # force an odd kernel size
    if method == "none" or k < 3:
        return img.copy()
    if method == "average":
        return cv2.blur(img, (k, k))
    if method == "median":
        return cv2.medianBlur(img, k)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (k, k), 0)
    raise ValueError(f"unknown smoothing method: {method!r}")


# ---------------------------------------------------------------------------
# Scoring against ground truth
# ---------------------------------------------------------------------------

def boundary_scores(pred_mask, gt_mask, tolerance=3):
    """Precision, recall and F1 of an edge map against a ground-truth mask.

    The ground-truth masks mark the whole road / river *area* (filled,
    white), but edge detectors output thin *boundaries*. A pixel-wise
    comparison of a 1 px line with a filled region would mean nothing, so
    we score against the mask's outline with a small spatial tolerance,
    which is the usual way edge detectors are evaluated (e.g. BSDS):

    1. GT boundary = mask pixels that touch the background
       (mask minus its 1 px erosion).
    2. Precision = share of predicted edge pixels within `tolerance` px of
       the GT boundary. "Of what I detected, how much is a real edge?"
    3. Recall = share of GT boundary pixels within `tolerance` px of a
       predicted edge. "Of the real edges, how much did I find?"
    4. F1 = 2PR / (P + R).

    The tolerance absorbs the 1-3 px misalignment between a hand-drawn
    mask and the actual edge in the image. Returns (precision, recall, f1).

    Known bias: thick edge maps (Sobel) are not penalised for thickness,
    because every pixel of a thick band near the boundary counts as
    correct, while the band covers more boundary and so raises recall.
    Compare thin maps (Canny vs Canny + post-processing) for the fairest
    reading. Benchmarks such as BSDS avoid this with one-to-one pixel
    matching, which is beyond the scope here.
    """
    pred = pred_mask > 0
    gt = (gt_mask > 0).astype(np.uint8)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")

    # The default erode border value counts outside-image pixels as
    # foreground, so a mask touching the tile border is not marked there.
    gt_boundary = (gt - cv2.erode(gt, np.ones((3, 3), np.uint8))) > 0

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (2 * tolerance + 1, 2 * tolerance + 1))
    near_gt = cv2.dilate(gt_boundary.astype(np.uint8), disc) > 0
    near_pred = cv2.dilate(pred.astype(np.uint8), disc) > 0

    n_pred, n_gt = pred.sum(), gt_boundary.sum()
    precision = (pred & near_gt).sum() / n_pred if n_pred else 0.0
    recall = (gt_boundary & near_pred).sum() / n_gt if n_gt else 0.0
    if precision + recall == 0:
        return float(precision), float(recall), 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def compute_f1(pred_mask, gt_mask, tolerance=3):
    """Boundary F1 score only (see boundary_scores)."""
    return boundary_scores(pred_mask, gt_mask, tolerance)[2]


# ---------------------------------------------------------------------------
# Visualisation helpers (pure, so they also work for report figures)
# ---------------------------------------------------------------------------

def signed_to_gray(arr):
    """Show a signed map (Gx or Gy) as gray: 128 = 0, darker = negative,
    brighter = positive. Scaled by the 99th percentile of |arr|."""
    scale = max(float(np.percentile(np.abs(arr), 99)), 1e-6)
    return np.clip(128 + arr / scale * 127, 0, 255).astype(np.uint8)


def magnitude_to_gray(magnitude):
    """Gradient magnitude as an 8-bit image, scaled by its 99th percentile."""
    scale = max(float(np.percentile(magnitude, 99)), 1e-6)
    return np.clip(magnitude / scale * 255, 0, 255).astype(np.uint8)


def direction_to_color(magnitude, direction):
    """Colour-wheel view of the gradient (returns a BGR image).

    hue = gradient direction theta (full circle), brightness = magnitude.
    Flat areas, whose direction is meaningless, come out dark.
    """
    hue = (np.degrees(direction) % 360) / 2       # OpenCV hue range is 0..179
    hsv = np.stack([hue.astype(np.uint8),
                    np.full(hue.shape, 255, np.uint8),
                    magnitude_to_gray(magnitude)], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def direction_color_wheel(size=120):
    """Legend for direction_to_color (BGR image): the hue at angle a around
    the centre is the colour of a gradient pointing in direction a, with
    screen y pointing down, like in the image."""
    c = (size - 1) / 2
    y, x = np.mgrid[0:size, 0:size] - c
    radius = np.sqrt(x ** 2 + y ** 2) / c
    hue = (np.degrees(np.arctan2(y, x)) % 360) / 2
    value = np.where(radius <= 1, radius * 255, 0)
    hsv = np.stack([hue.astype(np.uint8),
                    np.full(hue.shape, 255, np.uint8),
                    value.astype(np.uint8)], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
