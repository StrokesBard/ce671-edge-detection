# Edge detection for road & river identification

CE671 Introduction to Remote Sensing, term project.

A small Streamlit app that compares generic edge detectors (Sobel, Canny)
with Canny plus a feature-specific post-processing step, and scores each
against ground truth:

| Feature | Method 3 | Post-processing idea |
|---|---|---|
| Roads | Canny + Hough | Probabilistic Hough line transform keeps only long straight segments |
| Rivers | Canny + Morphology | Morphological closing bridges broken bank segments, then short fragments are removed |

For rivers, the edge detectors can run on **NDWI** (Normalized
Difference Water Index, from the green and near-infrared bands) instead of
true-colour brightness. This is standard practice in water mapping, and
the app compares the two inputs side by side.

Two more panels tie back to the lectures. One shows edge magnitude and
direction from Sobel. The other tests median vs Gaussian smoothing on
salt-and-pepper noise.

There is no machine learning or training. All methods are classical,
rule-based algorithms. Ground-truth masks are used only to score the
output, and to choose default parameter values on a separate tune set
(see [Evaluation](#evaluation-tune-then-test)).

## Setup and run

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

## Project structure

```
app.py                  Streamlit UI only (input handling, widgets, layout)
edge_methods.py         All image processing: pure functions, no Streamlit
evaluate.py             Tune-then-test evaluation (roads, rivers, rivers on NDWI)
requirements.txt
sample_images/
  roads/                2 DeepGlobe tiles: <id>_sat.jpg + <id>_mask.png
  rivers/               2 SWED patches: swed_<row>_<col>_sat.png, _ndwi.png,
                        _mask.png, and convert_swed.py, which makes them
DeepGlobe Road Extraction Dataset/
                        Full road dataset, used by evaluate.py (not by the app)
SWED_sample/            SWED water dataset (.npy), used by evaluate.py and
                        convert_swed.py (not by the app)
SWED_sample_useable/    Copies of the 25 SWED patches that contain a river
                        channel, ready to upload in the app, plus overview.png
```

Keep the folder names `sample_images/roads` and `sample_images/rivers`
as they are; the app looks them up by name.

`edge_methods.py` can be used directly from a notebook or script for report
figures:

```python
import cv2, edge_methods as em

color, gray = em.prepare_image(cv2.imread("sample_images/roads/613155_sat.jpg"))
gt = em.prepare_mask(cv2.imread("sample_images/roads/613155_mask.png"), gray.shape)
g = em.smooth_image(gray, "gaussian", 5)
print(em.boundary_scores(em.canny_hough_roads(g, 60, 80), gt))  # (P, R, F1)
```

## How the app works

1. **Input.** Every image (bundled sample or upload) goes through
   `prepare_image`: downscale so the longer side is at most 512 px, then
   convert to grayscale. Masks go through `prepare_mask` (resize with
   nearest neighbour, threshold at 127, white = feature). For SWED river
   data the sidebar also offers **NDWI** as the input (default), see below.
2. **Smoothing.** A 5x5 Gaussian blur is applied before every method.
   This is Canny's step 1; `cv2.Canny` does not smooth internally.
3. **Method.** Sobel (Otsu threshold on the gradient magnitude), Canny,
   or Canny plus the feature-specific step. The sidebar only shows the
   sliders for the selected method. Defaults depend on the feature and the
   input, and come from `evaluate.py`.
4. **Score.** A boundary F1 score (see below). For method 3 the app also
   shows the change against plain Canny with the same thresholds, which
   isolates the effect of the post-processing step.

### NDWI input (rivers)

```
NDWI = (Green - NIR) / (Green + NIR)        Sentinel-2: Green = B3, NIR = B8
```

Water reflects some green light but absorbs almost all near-infrared, so
NDWI is positive over water. Vegetation reflects NIR strongly, so its NDWI
is well below zero, and soil and buildings fall in between. In the NDWI
image the water-land boundary is the strongest edge, while the differences
between fields (which all reflect NIR) are squeezed together. In
true-colour grayscale, turbid water and green fields have similar
brightness, so river banks are no stronger than field boundaries.

`edge_methods.ndwi` computes the index and `ndwi_to_gray` maps it to 8
bits with a fixed scale (−1 → 0, 0 → 128, +1 → 255), so the same gray
level means the same NDWI in every patch. NDWI needs the near-infrared
band, so it is available for SWED data only: the bundled patches and any
SWED `.npy` patch you upload. Ordinary JPG and PNG uploads have only red,
green and blue.

### Scoring: boundary F1 with tolerance

The masks mark the filled road or river *area*, but edge detectors output
thin *boundaries*, so a pixel-wise F1 against the filled mask would be
meaningless. Instead (`edge_methods.boundary_scores`):

- the GT boundary is the outline of the mask;
- precision is the share of detected edge pixels within 3 px of the GT
  boundary;
- recall is the share of GT boundary pixels within 3 px of a detected
  edge;
- F1 = 2PR / (P + R).

F1 was chosen over IoU because precision and recall explain *why* a step
helps or not. For example, Hough raises precision by dropping clutter.

**Known bias:** thick edge maps are not penalised for thickness, so
Sobel's recall is inflated. Canny vs Canny + post-processing (both 1 px
wide) is the fair comparison.

### Analysis panels

- **True Colour vs NDWI** (rivers, SWED data only): runs all three methods
  on both inputs, each with its own tuned defaults, and shows the F1
  change NDWI brings for each method.
- **Edge Strength & Direction (Sobel):** Gx, Gy, magnitude
  `sqrt(Gx² + Gy²)` and direction `atan2(Gy, Gx)` shown on a colour wheel
  (hue = direction, brightness = strength). It works on the current input,
  independent of the selected method.
- **Noise Robustness:** adds salt-and-pepper noise (fixed seed) to the
  current input, then runs the selected method after each smoothing method
  (none, average, median, Gaussian), with F1 for each when ground truth is
  available.

## Sample data

### Roads: DeepGlobe Road Extraction (bundled)

Source: DeepGlobe 2018 Road Extraction Challenge (Demir et al., CVPR
Workshops 2018), available on Kaggle as "DeepGlobe Road Extraction
Dataset". Tiles are 1024 x 1024 px RGB at 50 cm/px. Masks are binary PNGs
with road = white. The full dataset is in
`DeepGlobe Road Extraction Dataset/`; only `train/` has masks.

Bundled tiles: `613155` (rural field grid) and `659689` (village with tree
cover). **They are best cases, not typical tiles.** They were picked from
a list of random tiles ranked by how much the Hough step improved F1, and
then by eye for long straight roads (the assumption Hough makes). Use them
to *show* what the step does. Use the test-set numbers below to *claim*
how much it helps. To swap in another tile, copy any
`DeepGlobe Road Extraction Dataset/train/<id>_sat.jpg` + `<id>_mask.png`
pair into `sample_images/roads/`. The app picks up every `*_sat.*` file
that has a matching `*_mask.png`.

### Rivers: SWED (bundled)

Source: Sentinel-2 Water Edges Dataset (SWED), UK Hydrographic Office,
https://openmldata.ukho.gov.uk/. The sample in `SWED_sample/` is one
Sentinel-2 L2A scene (tile T30UYC, 9 April 2017), which covers the Thames
estuary and the Essex coast. It is cut into 42 x 42 patches of 256 x 256 px
at 10 m/px.

- Images: `train/images/image_<row>_<col>.npy`, int16 arrays of shape
  (256, 256, 12). These are the 12 Sentinel-2 bands B1–B12 without B10,
  as reflectance x 10 000.
- Labels: `train/labels/chip_<row>_<col>.npy`, arrays of shape
  (1, 256, 256), with 1 = water.

The train files originally all started with the scene ID
`S2A_MSIL2A_20170409T105651_N0204_R094_T30UYC_20170409T110529_`
(Sentinel-2A, Level-2A, 9 April 2017, tile T30UYC). It was removed to keep
the names short, and the scripts accept both forms. The two files in
`test/` come from a different scene (15 February 2020, tile T36SXA), keep
their full names, and are not used by the project.

`sample_images/rivers/convert_swed.py` turns a patch into three PNGs:
true colour (bands B4/B3/B2, 2–98% stretch, `edge_methods.sentinel2_to_bgr`),
NDWI, and the mask.

```bash
python sample_images/rivers/convert_swed.py 39_17
```

You can also skip the conversion: choose "Upload your own" and upload a
SWED image `.npy`, plus its label `.npy` as the mask.

Most SWED patches are farmland or open sea, which makes for a poor river
demo: a patch with almost no water gives a nearly empty mask and an F1
close to zero. `SWED_sample_useable/` holds copies of only the 25 patches
where the water forms a channel (the same `river_like` rule `evaluate.py`
uses), as `images/image_<row>_<col>.npy` and `labels/chip_<row>_<col>.npy`.
`overview.png` shows all 25 with their IDs and the labelled water outlined.
Good demos: `35_1`, `35_2`, `35_4` (Thames, London), `39_17` (meander
through a town), `28_23` (wide rural estuary).

Bundled patches:

- `swed_35_3`: the Thames through London (Greenwich peninsula), urban banks.
- `swed_30_24`: a rural estuary channel, turbid water against marsh and
  fields.

They were picked by eye from the 25 patches where the water forms a
channel rather than open sea (see `river_like` in `evaluate.py`). To use a
different patch, convert it and delete one of the current sets, so the
folder keeps two.

### Rivers: other imagery

For a different area or a higher resolution:

1. **Imagery:** for example Sentinel-2 from the Copernicus Browser, or
   aerial or Google Earth imagery at 0.5–2 m.
2. **Mask, hand-digitised in QGIS:**
   - Open the image (Layer → Add Raster Layer).
   - Create a polygon layer (Layer → Create Layer → New Shapefile Layer,
     same CRS as the image) and trace the water surface.
   - Rasterize it with Raster → Conversion → Rasterize (Vector to Raster):
     burn value 255, output extent = the image layer, pixel size = the
     image's pixel size, data type Byte, nodata 0.
   - Export as PNG (right-click the layer → Export → Save As, format PNG).
3. Save both as `<name>_sat.jpg` (or `.png`) and `<name>_mask.png` in
   `sample_images/rivers/`. The mask must cover the same footprint as the
   image; its resolution may differ, because it is resized to match. NDWI
   is only available if you also add a `<name>_ndwi.png`.

## Results

All values are boundary F1 with 3 px tolerance, after 5x5 Gaussian
pre-smoothing.

### Evaluation (tune, then test)

```bash
python evaluate.py roads
```

```bash
python evaluate.py rivers
```

```bash
python evaluate.py rivers --input ndwi
```

Each run collects tiles, excluding the bundled samples, and splits them
into *tune* and *test* tiles. Each method's parameters are chosen by grid
search on the tune tiles only, and the feature-specific step reuses the
thresholds chosen for plain Canny. The chosen values are the app's
defaults. The scores below are on the test tiles, which were never used
for tuning.

**Roads** (about 3 min): random DeepGlobe train tiles with 3–15% road
cover, 60 tune and 200 test. Chosen: Sobel 7x7, Canny 60/80, Hough
threshold 50, min length 20 px, max gap 20 px. Random edges would score a
precision of about 0.13.

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Sobel | 0.218 | 0.834 | 0.338 |
| Canny | 0.225 | 0.793 | 0.340 |
| Canny + Hough | 0.261 | 0.788 | **0.359** |

Canny + Hough beat plain Canny on 70% of test tiles, by +0.019 F1 on
average (median +0.013). The gain is consistent but modest, and comes from
precision: the Hough step drops short and curved clutter. It cannot drop
building outlines, which are straight too, and it loses curved roads.

**Rivers** (about 1 min each): the 23 river-like SWED patches (25 minus
the 2 bundled ones), 10 tune and 13 test. Chosen for true colour: Sobel
3x3, Canny 5/40, closing 3 px. Chosen for NDWI: Sobel 3x3, Canny 80/200,
closing 5 px. Random edges would score a precision of about 0.14.

| Method | True colour P / R / F1 | NDWI P / R / F1 | F1 change |
|---|---|---|---|
| Sobel | 0.133 / 0.456 / 0.199 | 0.790 / 0.874 / **0.808** | +0.609 |
| Canny | 0.155 / 0.976 / 0.264 | 0.764 / 0.863 / **0.804** | +0.540 |
| Canny + Morphology | 0.154 / 0.927 / 0.260 | 0.775 / 0.804 / **0.783** | +0.523 |

**NDWI is the decisive step for rivers.** Every method roughly triples its
F1, and precision rises from barely above random (0.15 vs 0.14) to about
0.77. In true colour, field boundaries and urban texture produce edges as
strong as river banks, and no edge detector can tell them apart. With
NDWI, the water-land boundary dominates the image.

**The morphology step does not help on this data, on either input.** It
scores −0.004 F1 against plain Canny on true colour and −0.021 on NDWI
(better on 54% and 31% of test tiles). Recall shows why: at 10 m, Canny
already traces 86–98% of the bank pixels, so there are few gaps to
bridge, and closing also glues nearby clutter together. Closing should
matter more on high-resolution imagery (about 0.5–1 m), where trees,
shadows and boats break up the banks.

All river-like patches come from one Sentinel-2 scene, so the test set is
small and spatially close to the tune set. Treat the river numbers as
indicative.

### Bundled samples at default settings

| Sample | Input | Sobel | Canny | Canny + feature step |
|---|---|---|---|---|
| Road 613155 | true colour | 0.520 | 0.559 | **0.685** (Hough) |
| Road 659689 | true colour | 0.457 | 0.441 | **0.541** (Hough) |
| River swed_35_3 | true colour | 0.191 | 0.231 | 0.187 (Morphology) |
| River swed_35_3 | NDWI | 0.649 | **0.798** | 0.748 (Morphology) |
| River swed_30_24 | true colour | 0.111 | 0.255 | 0.261 (Morphology) |
| River swed_30_24 | NDWI | 0.975 | **0.978** | **0.978** (Morphology) |

The road tiles are best cases (see above), so their gains are larger than
the test-set average.

### Noise robustness

5% salt-and-pepper noise, 5x5 kernels, method 3.

| Sample | Input | None | Average | Median | Gaussian |
|---|---|---|---|---|---|
| Road 613155 | true colour | 0.137 | **0.715** | 0.696 | 0.202 |
| Road 659689 | true colour | 0.191 | 0.540 | **0.615** | 0.223 |
| River swed_35_3 | NDWI | 0.289 | **0.916** | 0.660 | 0.604 |
| River swed_30_24 | NDWI | 0.334 | **0.960** | 0.799 | 0.786 |

- **Median beat Gaussian on all four**, which is the lecture claim. The
  margin is wide on roads and smaller on the NDWI river patches.
- **The plain average filter often did best.** It won 3 of 4, including
  both river patches. Averaging a 5x5 window dilutes each impulse 25 times,
  which pushes it below Canny's thresholds. The median removes impulses
  completely, but it also deletes any feature narrower than about half
  its window, which at 10 m includes small creeks. That is a likely reason
  for the gap, though it was not measured separately. Either way, "median
  is the best choice for salt-and-pepper noise" does not hold in every
  case.
- **A 5x5 Gaussian** (sigma ≈ 1.1) barely beats no smoothing, because its
  weights are concentrated near the centre pixel, so an impulse stays
  almost intact.
- On **true-colour** river patches the test is uninformative: F1 is near
  random with or without noise.
