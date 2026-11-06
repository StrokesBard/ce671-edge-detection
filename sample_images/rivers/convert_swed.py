"""
Converts SWED patches (Sentinel-2 Water Edges Dataset, UK Hydrographic
Office, https://openmldata.ukho.gov.uk/) from .npy into the PNG pairs the
app reads, saved next to this script:
    swed_<row>_<col>_sat.png    true colour (bands B4, B3, B2)
    swed_<row>_<col>_ndwi.png   NDWI from bands B3 and B8, water bright
    swed_<row>_<col>_mask.png   water = white

SWED images are (256, 256, 12) int16 arrays of Sentinel-2 L2A reflectance
at 10 m/px; labels are (1, 256, 256) arrays with 1 = water. A patch is
named by its row_col position in the scene: image_35_3.npy with label
chip_35_3.npy (the original scene-prefixed names work too).

    python sample_images/rivers/convert_swed.py               # bundled patches
    python sample_images/rivers/convert_swed.py 39_17 28_23   # any others
"""

import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
import edge_methods as em  # noqa: E402

SWED = ROOT / "SWED_sample" / "train"
BUNDLED = ["35_3", "30_24"]   # Thames in London; a rural estuary channel


def convert(patch):
    images = list((SWED / "images").glob(f"*image_{patch}.npy"))
    if len(images) != 1:
        sys.exit(f"patch {patch}: expected 1 image in {SWED / 'images'}, "
                 f"found {len(images)}")
    label = SWED / "labels" / re.sub(r"image_(\d+_\d+)\.npy$", r"chip_\1.npy",
                                     images[0].name)
    bands, water = np.load(images[0]), np.load(label)[0]
    cv2.imwrite(str(HERE / f"swed_{patch}_sat.png"), em.sentinel2_to_bgr(bands))
    cv2.imwrite(str(HERE / f"swed_{patch}_ndwi.png"),
                em.ndwi_to_gray(em.ndwi(bands)))
    cv2.imwrite(str(HERE / f"swed_{patch}_mask.png"),
                np.where(water > 0, 255, 0).astype(np.uint8))
    print(f"wrote swed_{patch}_sat.png, _ndwi.png, _mask.png")


if __name__ == "__main__":
    for p in sys.argv[1:] or BUNDLED:
        convert(p)
