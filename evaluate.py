"""
evaluate.py
===========
Tune-then-test evaluation of the three methods for one feature type, so
the reported scores are not measured on the tiles the parameters were
chosen on.

    python evaluate.py roads                  # DeepGlobe tiles, ~3 min
    python evaluate.py rivers                 # SWED patches, true colour
    python evaluate.py rivers --input ndwi    # SWED patches, NDWI

1. Collect tiles, excluding the bundled sample tiles:
     roads   random DeepGlobe train tiles with 3-15% road cover
     rivers  SWED patches where the water forms a channel rather than open
             sea: 4-35% water, no water pixel more than 30 px (300 m) from
             a bank, and the largest water body spans at least 200 of the
             256 px
2. Shuffle them and split into a tune set and a test set.
3. On the tune set, choose each method's parameters by grid search (best
   mean boundary F1):
     Sobel               kernel size
     Canny               low / high threshold
     Canny + Hough       Hough threshold, min line length, max line gap
     Canny + Morphology  closing kernel size
   The feature-specific step reuses the thresholds chosen for plain Canny,
   so the comparison isolates that step.
4. Score every method on the test set with those parameters.

No model is trained. This only chooses parameter values, the way you would
by hand with the sliders, but on data the final scores never see.
"""

import argparse
import itertools
import random
import re
from pathlib import Path

import cv2
import numpy as np

import edge_methods as em

ROOT = Path(__file__).parent
DEEPGLOBE = ROOT / "DeepGlobe Road Extraction Dataset" / "train"
SWED = ROOT / "SWED_sample" / "train"
PRESMOOTH = ("gaussian", 5)                   # same as app.py

SOBEL_GRID = (3, 5, 7)
CANNY_GRID = [(lo, hi) for lo in (5, 10, 20, 30, 40, 50, 60, 80)
              for hi in (40, 60, 80, 100, 125, 150, 200, 250) if hi > lo]
HOUGH_GRID = [{"threshold": t, "min_line_length": mll, "max_line_gap": gap}
              for t, mll, gap in itertools.product((30, 50, 80, 120),
                                                   (20, 30, 50, 80),
                                                   (10, 20, 30))]
KERNEL_GRID = (3, 5, 7, 9, 11, 15)

FEATURES = {
    # name of method 3, its parameter grid, detector factory, default split
    "roads": ("Canny + Hough", HOUGH_GRID,
              lambda lo, hi, h: lambda g: em.canny_hough_roads(g, lo, hi, h),
              (60, 200)),
    "rivers": ("Canny + Morphology", KERNEL_GRID,
               lambda lo, hi, k: lambda g: em.canny_morph_rivers(g, lo, hi, k),
               (10, None)),
}


def sample_names(feature):
    return {p.name.rsplit("_sat", 1)[0]
            for p in (ROOT / "sample_images" / feature).glob("*_sat.*")}


def tile(bgr, mask):
    """(smoothed gray, mask) preprocessed exactly like the app."""
    _, gray = em.prepare_image(bgr)
    return em.smooth_image(gray, *PRESMOOTH), em.prepare_mask(mask, gray.shape)


def load_roads(n, seed, min_cover=0.03, max_cover=0.15):
    """n random DeepGlobe tiles with min_cover..max_cover road pixels."""
    exclude = sample_names("roads")
    masks = sorted(DEEPGLOBE.glob("*_mask.png"))
    random.Random(seed).shuffle(masks)
    tiles = []
    for m in masks:
        tile_id = m.name.split("_")[0]
        if tile_id in exclude:
            continue
        mask = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        if not min_cover <= (mask > 127).mean() <= max_cover:
            continue
        tiles.append(tile(cv2.imread(str(m.with_name(f"{tile_id}_sat.jpg"))),
                          mask))
        if len(tiles) == n:
            break
    return tiles


def river_like(water):
    """True if the water in a SWED label forms a channel, not open sea."""
    if not 0.04 <= water.mean() <= 0.35:
        return False
    if cv2.distanceTransform(water, cv2.DIST_L2, 5).max() > 30:
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats(water, connectivity=8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return max(stats[largest, cv2.CC_STAT_WIDTH],
               stats[largest, cv2.CC_STAT_HEIGHT]) >= 200


def load_rivers(n, seed, use_ndwi=False):
    """Up to n river-like SWED patches, in random order, as true-colour or
    NDWI images."""
    exclude = sample_names("rivers")
    tiles = []
    for img in sorted((SWED / "images").glob("*.npy")):
        # Works for image_3_19.npy and for the original long scene-prefixed name
        patch = re.search(r"image_(\d+_\d+)\.npy$", img.name).group(1)
        if f"swed_{patch}" in exclude:
            continue
        label = SWED / "labels" / re.sub(r"image_(\d+_\d+)\.npy$",
                                         r"chip_\1.npy", img.name)
        water = (np.load(label)[0] > 0).astype(np.uint8)
        if river_like(water):
            bands = np.load(img)
            image = (em.ndwi_to_gray(em.ndwi(bands)) if use_ndwi
                     else em.sentinel2_to_bgr(bands))
            tiles.append(tile(image, water * 255))
    random.Random(seed).shuffle(tiles)
    return tiles[:n]


def scores(tiles, detect):
    """(n_tiles, 3) array of precision, recall, F1."""
    return np.array([em.boundary_scores(detect(g), gt) for g, gt in tiles])


def best(tiles, candidates, make_detect):
    """Candidate with the highest mean F1 on `tiles`."""
    return max(candidates,
               key=lambda c: scores(tiles, make_detect(c))[:, 2].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("feature", choices=FEATURES)
    ap.add_argument("--tune", type=int,
                    help="tiles in the tune set (default: roads 60, rivers 10)")
    ap.add_argument("--test", type=int,
                    help="tiles in the test set (default: roads 200, "
                         "rivers all remaining)")
    ap.add_argument("--input", choices=["rgb", "ndwi"], default="rgb",
                    help="rivers only: true-colour grayscale or NDWI")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.feature == "roads" and args.input == "ndwi":
        ap.error("NDWI needs Sentinel-2 bands; it is only available for rivers")

    third_name, third_grid, third_factory, (n_tune, n_test) = FEATURES[args.feature]
    n_tune = args.tune or n_tune
    n_test = args.test or n_test
    if args.feature == "roads":
        tiles = load_roads(n_tune + n_test, args.seed)
    else:
        tiles = load_rivers(n_tune + n_test if n_test else None, args.seed,
                            args.input == "ndwi")
    tune, test = tiles[:n_tune], tiles[n_tune:]
    print(f"{args.feature} ({args.input}): {len(tune)} tune tiles, "
          f"{len(test)} test tiles (seed {args.seed})")

    ksize = best(tune, SOBEL_GRID, lambda k: lambda g: em.sobel_edge(g, k))
    low, high = best(tune, CANNY_GRID,
                     lambda c: lambda g: em.canny_edge(g, *c))
    third = best(tune, third_grid, lambda p: third_factory(low, high, p))
    print(f"\nChosen on the tune set: Sobel ksize={ksize}, Canny {low}/{high}, "
          f"{third_name} {third}")

    results = {
        "Sobel": scores(test, lambda g: em.sobel_edge(g, ksize)),
        "Canny": scores(test, lambda g: em.canny_edge(g, low, high)),
        third_name: scores(test, third_factory(low, high, third)),
    }
    print(f"\nTest set ({len(test)} tiles), mean over tiles:")
    print(f"  {'method':<20}{'precision':>10}{'recall':>8}{'F1':>8}")
    for name, s in results.items():
        p, r, f = s.mean(axis=0)
        print(f"  {name:<20}{p:>10.3f}{r:>8.3f}{f:>8.3f}")

    gain = results[third_name][:, 2] - results["Canny"][:, 2]
    print(f"\n{third_name} vs Canny: mean F1 change {gain.mean():+.3f}, "
          f"median {np.median(gain):+.3f}, better on {(gain > 0).mean():.0%} "
          f"of test tiles")

    if args.feature == "rivers":
        # Ablation: a 1 px kernel makes closing a no-op, leaving only the
        # short-fragment filter. Shows what the closing itself contributes.
        no_close = scores(test, third_factory(low, high, 1))[:, 2].mean()
        print(f"Same pipeline without closing (fragment filter only): "
              f"F1 {no_close:.3f}")


if __name__ == "__main__":
    main()
