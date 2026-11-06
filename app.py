"""
app.py - Streamlit UI for the CE671 term project
"Edge detection for road and river identification in satellite imagery".

All image processing lives in edge_methods.py; this file only handles input,
widgets and layout.

Run:  streamlit run app.py
"""

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import edge_methods as em

SAMPLE_DIR = Path(__file__).parent / "sample_images"
TOLERANCE = 3                  # px tolerance for the boundary F1 score
PRESMOOTH = ("gaussian", 5)    # Canny step 1; applied before every method
THIRD_METHOD = {"Roads": "Canny + Hough", "Rivers": "Canny + Morphology"}
# Defaults per (feature, input), chosen on the tune split by evaluate.py
DEFAULTS = {
    ("Roads", "True colour"): {"ksize": 7, "canny": (60, 80)},
    ("Rivers", "True colour"): {"ksize": 3, "canny": (5, 40), "kernel": 3},
    ("Rivers", "NDWI"): {"ksize": 3, "canny": (80, 200), "kernel": 5},
}
SAMPLE_NOTES = {
    "Roads": "DeepGlobe Road Extraction tile (50 cm/px, 1024 px), "
             "processed at 512 px.",
    "Rivers": "SWED Sentinel-2 patch (10 m/px, 256 px), Thames estuary "
              "area. Name = row_col of the patch in the scene.",
}
NDWI_HELP = ("NDWI = (Green - NIR) / (Green + NIR), from Sentinel-2 bands B3 "
             "and B8. Water absorbs near-infrared, so it comes out bright and "
             "land dark. Needs the NIR band, so only SWED data has it.")


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def find_samples(feature):
    """{display name: (image, mask, ndwi)} for DeepGlobe-style pairs
    <name>_sat.<ext> + <name>_mask.png in the feature's sample folder.
    <name>_ndwi.png is optional (SWED patches only)."""
    pairs = {}
    for sat in sorted((SAMPLE_DIR / feature.lower()).glob("*_sat.*")):
        name = sat.stem[:-len("_sat")]
        mask = sat.with_name(f"{name}_mask.png")
        ndwi = sat.with_name(f"{name}_ndwi.png")
        pairs[name] = (sat, mask if mask.exists() else None,
                       ndwi if ndwi.exists() else None)
    return pairs


@st.cache_data
def read_sample(img_path, mask_path, ndwi_path):
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_path else None)
    ndwi = (cv2.imread(str(ndwi_path), cv2.IMREAD_GRAYSCALE)
            if ndwi_path else None)
    return bgr, mask, ndwi


def decode_upload(uploaded, flags):
    return cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), flags)


def load_npy(uploaded):
    """Array from an uploaded .npy file (pickled objects refused), or None."""
    try:
        return np.load(io.BytesIO(uploaded.getvalue()), allow_pickle=False)
    except (ValueError, OSError, EOFError):
        return None


def is_npy(uploaded):
    return uploaded is not None and uploaded.name.lower().endswith(".npy")


def detect(method, img, feature, p):
    """Run the selected method (0 = Sobel, 1 = Canny, 2 = feature-specific)."""
    if method == 0:
        return em.sobel_edge(img, p["ksize"])
    if method == 1:
        return em.canny_edge(img, p["low"], p["high"])
    if feature == "Roads":
        return em.canny_hough_roads(img, p["low"], p["high"],
                                    {"threshold": p["hough_threshold"]})
    return em.canny_morph_rivers(img, p["low"], p["high"], p["kernel"])


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CE671 Edge Detection", layout="wide")
st.title("Edge detection for road & river identification")
st.caption("CE671 Introduction to Remote Sensing, term project. Generic edge "
           "detectors (Sobel, Canny) vs Canny with a feature-specific "
           "post-processing step, scored against ground truth.")

top = st.columns([1, 2])
feature = top[0].radio("Feature type", ["Roads", "Rivers"], horizontal=True)
method_names = ["Sobel", "Canny", THIRD_METHOD[feature]]
# One radio per feature type (its labels differ), with the selected index
# carried over so e.g. "Canny + Hough" becomes "Canny + Morphology" on switch.
method = top[1].radio("Method", [0, 1, 2], format_func=lambda i: method_names[i],
                      index=st.session_state.get("method_idx", 0),
                      horizontal=True, key=f"method_{feature}")
st.session_state["method_idx"] = method

with st.sidebar:
    st.header("Input image")
    source = st.radio("Image source", ["Use sample image", "Upload your own"])
    if source == "Use sample image":
        samples = find_samples(feature)
        if not samples:
            st.error(f"No sample images found in sample_images/{feature.lower()}/")
            st.stop()
        sample_name = st.selectbox("Sample", list(samples))
        st.caption(SAMPLE_NOTES[feature])
        has_ndwi = samples[sample_name][2] is not None
    else:
        upload = st.file_uploader(
            "Satellite image", type=["jpg", "jpeg", "png", "npy"],
            help="JPG/PNG, or a SWED .npy patch (12 Sentinel-2 bands), which "
                 "also makes NDWI available.")
        upload_gt = st.file_uploader(
            "Ground-truth mask (optional, for scoring)",
            type=["png", "jpg", "jpeg", "npy"],
            help="Binary mask covering the same area as the image: white (or "
                 f"1 in a .npy label) = {feature.lower()[:-1]}, black/0 = "
                 "background.")
        has_ndwi = is_npy(upload)

    input_name = "True colour"
    if has_ndwi and feature == "Rivers":
        input_name = st.radio("Edge detection input", ["NDWI", "True colour"],
                              horizontal=True, help=NDWI_HELP)
    defaults = DEFAULTS[(feature, input_name)]

    st.header("Parameters")
    params = {}
    if method == 0:
        params["ksize"] = st.select_slider("Sobel kernel size", [3, 5, 7],
                                           defaults["ksize"])
    else:
        d_low, d_high = defaults["canny"]
        params["low"] = st.slider("Canny low threshold", 0, 300, d_low)
        params["high"] = st.slider("Canny high threshold", 0, 300, d_high)
        if params["low"] >= params["high"]:
            st.caption("Low >= high: the two thresholds are swapped internally.")
    if method == 2 and feature == "Roads":
        params["hough_threshold"] = st.slider(
            "Hough threshold (votes)", 10, 200, em.DEFAULT_HOUGH["threshold"],
            help="Minimum number of collinear edge pixels for a line. Higher "
                 "= fewer, more certain lines. Min segment length "
                 f"({em.DEFAULT_HOUGH['min_line_length']} px) and max gap "
                 f"({em.DEFAULT_HOUGH['max_line_gap']} px) are fixed in "
                 "edge_methods.DEFAULT_HOUGH.")
    if method == 2 and feature == "Rivers":
        params["kernel"] = st.slider(
            "Closing kernel size (px)", 3, 21, defaults["kernel"], step=2,
            help="Diameter of the elliptical structuring element. Gaps "
                 "narrower than about this are bridged; too large glues "
                 "texture together.")
    st.caption(f"Every method runs on the image after a {PRESMOOTH[1]}x"
               f"{PRESMOOTH[1]} Gaussian blur (Canny's smoothing step).")

# --- Load the image (sample or upload) and run the shared preprocessing ---
ndwi_raw = None
if source == "Use sample image":
    bgr, gt_raw, ndwi_raw = read_sample(*samples[sample_name])
else:
    if upload is None:
        st.info("Upload a satellite image in the sidebar to begin.")
        st.stop()
    if is_npy(upload):
        bands = load_npy(upload)
        if bands is None or bands.ndim != 3 or bands.shape[2] < 8:
            st.error("Expected a SWED patch: an (H, W, 12) array of "
                     "Sentinel-2 bands.")
            st.stop()
        bgr = em.sentinel2_to_bgr(bands)
        ndwi_raw = em.ndwi_to_gray(em.ndwi(bands))
    else:
        bgr = decode_upload(upload, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read the uploaded image.")
            st.stop()
    gt_raw = None
    if upload_gt is not None:
        if is_npy(upload_gt):
            label = load_npy(upload_gt)
            if label is not None and np.squeeze(label).ndim == 2:
                gt_raw = np.where(np.squeeze(label) > 0, 255, 0).astype(np.uint8)
        else:
            gt_raw = decode_upload(upload_gt, cv2.IMREAD_GRAYSCALE)
        if gt_raw is None:
            st.warning("Could not read the ground-truth mask; scoring disabled.")

color, gray = em.prepare_image(bgr)
ndwi_gray = em.prepare_image(ndwi_raw)[1] if ndwi_raw is not None else None
gt = em.prepare_mask(gt_raw, gray.shape) if gt_raw is not None else None
if gt_raw is not None:
    aspect_img = bgr.shape[1] / bgr.shape[0]
    aspect_gt = gt_raw.shape[1] / gt_raw.shape[0]
    if abs(aspect_img - aspect_gt) > 0.01 * aspect_img:
        st.warning("The mask's aspect ratio differs from the image's; it was "
                   "stretched to fit, so scores may be off.")

# The image the edge detectors see: true-colour grayscale, or the NDWI band
base = ndwi_gray if input_name == "NDWI" else gray
on_input = " (on NDWI)" if input_name == "NDWI" else ""
smoothed = em.smooth_image(base, *PRESMOOTH)
edges = detect(method, smoothed, feature, params)

# --- Main comparison: original | edge map | ground truth ---
c1, c2, c3 = st.columns(3)
c1.image(color, channels="BGR", caption="Original", width="stretch")
c2.image(edges, caption=f"{method_names[method]} edge map{on_input}",
         width="stretch")
if gt is not None:
    c3.image(gt, caption=f"Ground truth (white = {feature.lower()})",
             width="stretch")
else:
    c3.markdown("**Ground truth**")
    c3.info("No ground truth provided.")

if gt is not None and st.toggle("Show F1 score", value=True):
    precision, recall, f1 = em.boundary_scores(edges, gt, TOLERANCE)
    delta = None
    if method == 2:
        base_f1 = em.compute_f1(em.canny_edge(smoothed, params["low"],
                                              params["high"]), gt, TOLERANCE)
        delta = f"{f1 - base_f1:+.3f} vs plain Canny (same thresholds)"
    m = st.columns(3)
    m[0].metric("F1", f"{f1:.3f}", delta=delta)
    m[1].metric("Precision", f"{precision:.3f}")
    m[2].metric("Recall", f"{recall:.3f}")
    st.caption(f"Scored against the outline of the ground-truth mask with a "
               f"±{TOLERANCE} px tolerance. Precision = share of detected "
               "edge pixels that lie on a true boundary; recall = share of "
               "the true boundary that was detected. Thick edge maps (Sobel) "
               "get a recall bonus from this metric, so Canny vs Canny + "
               "post-processing is the fairest comparison.")

st.divider()

# --- Analysis panel: true colour vs NDWI as the input (rivers, SWED only) ---
if feature == "Rivers" and ndwi_gray is not None:
    with st.expander("True Colour vs NDWI", expanded=True):
        st.markdown(
            "Water absorbs almost all near-infrared (NIR) light, while "
            "vegetation reflects it strongly. NDWI turns that into one band "
            "where water is bright and land is dark, so the water-land "
            "boundary becomes the strongest edge in the scene. In "
            "true-colour grayscale, turbid water and green fields have "
            "similar brightness, so river banks are no stronger than field "
            "boundaries. Each row runs all three methods on one input, with "
            "that input's own default parameters (tuned by evaluate.py), not "
            "the sidebar sliders.")
        st.latex(r"\mathrm{NDWI} = \frac{G - NIR}{G + NIR} \qquad "
                 r"G = \mathrm{B3},\; NIR = \mathrm{B8}")
        rgb_f1 = {}
        for name, img, caption in (
                ("True colour", gray, "True-colour grayscale input"),
                ("NDWI", ndwi_gray, "NDWI input (water bright)")):
            d = DEFAULTS[("Rivers", name)]
            p = {"ksize": d["ksize"], "low": d["canny"][0],
                 "high": d["canny"][1], "kernel": d["kernel"]}
            img_smoothed = em.smooth_image(img, *PRESMOOTH)
            cols = st.columns(4)
            cols[0].image(img, width="stretch", caption=caption)
            for m, col in enumerate(cols[1:]):
                e = detect(m, img_smoothed, "Rivers", p)
                col.image(e, width="stretch", caption=f"{method_names[m]}, {name}")
                if gt is not None:
                    score = em.compute_f1(e, gt, TOLERANCE)
                    delta = (f"{score - rgb_f1[m]:+.3f} vs true colour"
                             if name == "NDWI" else None)
                    rgb_f1[m] = score
                    col.metric("F1", f"{score:.3f}", delta=delta)

# --- Analysis panel: gradient magnitude and direction ---
with st.expander("Edge Strength & Direction (Sobel)"):
    st.markdown(f"Sobel gradients of the current input{on_input} (after the "
                "same Gaussian blur), independent of the method selected above.")
    dir_ksize = st.select_slider("Sobel kernel size ", [3, 5, 7], 3,
                                 key="dir_ksize")
    gx, gy = em.sobel_gradients(smoothed, dir_ksize)
    magnitude, direction = em.sobel_magnitude_direction(smoothed, dir_ksize)

    g = st.columns(4)
    g[0].image(em.signed_to_gray(gx), width="stretch",
               caption="Gx = dI/dx (responds to vertical edges)")
    g[1].image(em.signed_to_gray(gy), width="stretch",
               caption="Gy = dI/dy (responds to horizontal edges)")
    g[2].image(em.magnitude_to_gray(magnitude), width="stretch",
               caption="Edge strength (magnitude)")
    g[3].image(em.direction_to_color(magnitude, direction), channels="BGR",
               width="stretch",
               caption="Edge direction (hue), brightness = strength")

    legend, text = st.columns([1, 5])
    legend.image(em.direction_color_wheel(), channels="BGR",
                 caption="Direction key")
    with text:
        st.latex(r"M = \sqrt{G_x^2 + G_y^2} \qquad "
                 r"\theta = \operatorname{atan2}(G_y,\, G_x)")
        st.markdown(
            "In Gx and Gy, mid-gray = 0, bright = positive, dark = negative. "
            "θ is the gradient direction: it points across an edge from dark "
            "to bright, so the edge itself runs perpendicular to it (θ ± 90°). "
            "Both banks of a road or river have opposite colours in the "
            "direction map, because their gradients point in opposite "
            "directions. atan2 is used instead of arctan(Gy/Gx) because it "
            "never divides by zero and keeps the full 360° range.")

# --- Analysis panel: salt-and-pepper noise vs smoothing method ---
with st.expander("Noise Robustness"):
    st.markdown(
        f"Adds salt-and-pepper noise to the current input{on_input}, then "
        f"runs **{method_names[method]}** (with the parameters in the "
        "sidebar) after each smoothing method. This replaces the fixed "
        "Gaussian blur of the main comparison, to test the lecture claim that "
        "median filtering handles salt-and-pepper noise better than Gaussian "
        "smoothing.")
    n1, n2, n3 = st.columns(3)
    add_noise = n1.toggle("Add salt-and-pepper noise", value=True)
    amount = n1.slider("Noise amount (fraction of pixels)", 0.0, 0.30, 0.05,
                       0.01, disabled=not add_noise)
    chosen = n2.multiselect("Smoothing methods", em.SMOOTHING_METHODS,
                            default=list(em.SMOOTHING_METHODS))
    smooth_k = n3.select_slider("Smoothing kernel size", [3, 5, 7, 9, 11], 5)

    noisy = em.add_salt_pepper_noise(base, amount) if add_noise else base
    chosen = [m for m in em.SMOOTHING_METHODS if m in chosen]
    if not chosen:
        st.info("Select at least one smoothing method.")
    for col, smooth_method in zip(st.columns(max(len(chosen), 1)), chosen):
        smoothed_noisy = em.smooth_image(noisy, smooth_method, smooth_k)
        noisy_edges = detect(method, smoothed_noisy, feature, params)
        col.image(smoothed_noisy, width="stretch",
                  caption=("Noisy input" if smooth_method == "none"
                           else f"After {smooth_method} {smooth_k}x{smooth_k}"))
        col.image(noisy_edges, width="stretch",
                  caption=f"{method_names[method]} edges")
        if gt is not None:
            col.metric(f"F1, {smooth_method}",
                       f"{em.compute_f1(noisy_edges, gt, TOLERANCE):.3f}")
