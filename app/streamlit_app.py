"""ForenSort local review app.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from pathlib import Path
import os
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ebts import parse_lffs
from inference import load_model, predict_with_confidence, prepare_image, summarize_prediction

CHECKPOINT = ROOT / "outputs" / "checkpoints" / "baseline_best.pt"
SIZE = 512
QUALITY_COLORS = ["#161a2b", "#d64f5f", "#ef8b4d", "#e5c453", "#5dbb8a", "#79b8d8"]
QUALITY_NAMES = ["background / unusable", "very limited", "limited", "fair", "good", "clear"]

st.set_page_config(page_title="ForenSort | latent quality review", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap');
    :root { --ink:#172033; --muted:#667085; --paper:#f5f4ef; --line:#d8d9d1; --accent:#d64f5f; }
    .stApp { background:var(--paper); color:var(--ink); }
    .block-container { max-width:1180px; padding-top:3.5rem; padding-bottom:4rem; }
    h1,h2,h3,p,div,span,label { font-family:'Manrope', sans-serif; }
    h1 { letter-spacing:-.055em; font-size:clamp(2.2rem,5vw,4.7rem); line-height:.98; }
    h2 { letter-spacing:-.035em; }
    .eyebrow { color:var(--accent); font:500 .72rem 'DM Mono', monospace; letter-spacing:.12em; text-transform:uppercase; }
    .lede { color:var(--muted); max-width:600px; font-size:1.05rem; line-height:1.65; }
    .rule { border-top:1px solid var(--line); margin:1.5rem 0 2rem; }
    .metric-card { background:#fff; border:1px solid var(--line); padding:1rem 1.1rem; min-height:94px; }
    .metric-label { color:var(--muted); font:500 .7rem 'DM Mono', monospace; text-transform:uppercase; letter-spacing:.08em; }
    .metric-value { color:var(--ink); font-size:1.45rem; font-weight:800; margin-top:.35rem; }
    .note { color:var(--muted); font-size:.8rem; line-height:1.5; }
    [data-testid="stFileUploader"] { border:1px dashed #b8b9b0; background:rgba(255,255,255,.45); padding:.4rem; }
    button[kind="primary"] { background:var(--accent); border-color:var(--accent); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(CHECKPOINT, device), device


def make_overlay(image: np.ndarray, prediction: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, nw, nh = box
    heatmap = cv2.resize(prediction, (nw, nh), interpolation=cv2.INTER_NEAREST)
    palette = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in QUALITY_COLORS], dtype=np.uint8)
    color = palette[heatmap]
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    blended = (0.58 * base[y0:y0 + nh, x0:x0 + nw] + 0.42 * color).astype(np.uint8)
    result = base.copy()
    result[y0:y0 + nh, x0:x0 + nw] = blended
    return result


def content_mask(prediction_shape: tuple[int, int], box: tuple[int, int, int, int], size: int) -> np.ndarray:
    """Select output cells whose centre lies inside the non-padding image area."""
    x0, y0, nw, nh = box
    stride_y = size / prediction_shape[0]
    stride_x = size / prediction_shape[1]
    ys = (np.arange(prediction_shape[0]) + 0.5) * stride_y
    xs = (np.arange(prediction_shape[1]) + 0.5) * stride_x
    return (
        (ys[:, None] >= y0)
        & (ys[:, None] < y0 + nh)
        & (xs[None, :] >= x0)
        & (xs[None, :] < x0 + nw)
    )


def read_uploaded_image(uploaded) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Read a standard image or extract the embedded image from an NIST LFFS file."""
    if uploaded.name.lower().endswith(".lffs"):
        fd, temp_path = tempfile.mkstemp(suffix=".lffs")
        try:
            with os.fdopen(fd, "wb") as temp:
                temp.write(uploaded.getvalue())
            latent = parse_lffs(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
        if latent.image is None:
            raise ValueError("The LFFS file does not contain an embedded image")
        return latent.image, latent.quality, "NIST LFFS"
    return np.array(Image.open(uploaded).convert("L")), None, "standard image"


st.markdown('<div class="eyebrow">ForenSort / latent quality review</div>', unsafe_allow_html=True)
st.title("See where a latent fingerprint holds up.")
st.markdown(
    '<p class="lede">Upload a grayscale image or NIST LFFS transaction. The trained baseline marks regional ridge quality from 0 (weak or absent) to 5 (clear).</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Choose a fingerprint image or NIST LFFS file",
    type=["lffs", "png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)
if not uploaded_files:
    st.info("Start with a PNG, JPG, TIFF, or NIST LFFS file. The model is advisory only and does not identify a person.")
    st.stop()

model, device = get_model()
cases = []
for uploaded in uploaded_files:
    try:
        original, expert_quality, source_type = read_uploaded_image(uploaded)
        tensor, prepared, box = prepare_image(original, SIZE)
        prediction, confidence, entropy = predict_with_confidence(model, tensor, device)
        content = content_mask(prediction.shape, box, SIZE)
        summary = summarize_prediction(prediction, confidence, entropy, content)
        cases.append({
            "name": uploaded.name,
            "original": original,
            "expert_quality": expert_quality,
            "source_type": source_type,
            "prepared": prepared,
            "prediction": prediction,
            "confidence_map": confidence,
            "entropy": entropy,
            "box": box,
            "summary": summary,
        })
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Could not analyze {uploaded.name}: {exc}")

if not cases:
    st.stop()

cases.sort(key=lambda item: item["summary"]["image_score"], reverse=True)
st.markdown('<div class="eyebrow">Batch ranking</div>', unsafe_allow_html=True)
st.markdown("### Images sorted by advisory quality score")
st.caption("The score combines average regional quality and usable-region coverage. It is for review order, not a forensic conclusion.")
table = [
    {
        "Rank": i,
        "Image": case["name"],
        "Score": round(case["summary"]["image_score"], 1),
        "Average level": round(case["summary"]["average_quality"], 2),
        "High-quality regions": f"{case['summary']['high_quality_regions']:.1f}%",
        "Usable regions": f"{case['summary']['usable_regions']:.1f}%",
        "Category": case["summary"]["category"],
    }
    for i, case in enumerate(cases, 1)
]
st.dataframe(table, use_container_width=True, hide_index=True)

selected_name = st.selectbox("Inspect an image", [case["name"] for case in cases])
case = next(item for item in cases if item["name"] == selected_name)
original = case["original"]
expert_quality = case["expert_quality"]
source_type = case["source_type"]
prepared = case["prepared"]
prediction = case["prediction"]
confidence = case["confidence_map"]
entropy = case["entropy"]
box = case["box"]
summary = case["summary"]
image_area = prediction[content_mask(prediction.shape, box, SIZE)]
confidence_area = confidence[content_mask(confidence.shape, box, SIZE)]
mean_level = float(summary["average_quality"])
clear_share = float(summary["high_quality_regions"])
weak_share = float((image_area <= 1).mean() * 100)
dominant = int(np.bincount(image_area, minlength=6).argmax())
mean_confidence = float(summary["confidence"])
low_confidence_share = float((confidence_area < 0.55).mean() * 100)
mean_entropy = float(summary["uncertainty"])
overlay = make_overlay(prepared, prediction, box)
display_original = cv2.cvtColor(prepared, cv2.COLOR_GRAY2RGB)

left, right = st.columns([1.35, 1], gap="large")
with left:
    st.markdown('<div class="eyebrow">Regional map</div>', unsafe_allow_html=True)
    st.image(overlay, caption="Model overlay — warmer colors indicate lower predicted quality.", use_container_width=True)
with right:
    st.markdown('<div class="eyebrow">Reading the image</div>', unsafe_allow_html=True)
    st.markdown(f"### Most common level: {dominant} · {QUALITY_NAMES[dominant]}")
    st.markdown(f"**Advisory image category:** {summary['category']} · score {summary['image_score']:.1f}/100")
    st.markdown(
        '<p class="note">The map is predicted at a regional resolution and upscaled for viewing. It is a quality aid, not an identification or forensic conclusion.</p>',
        unsafe_allow_html=True,
    )
    if float(original.std()) < 12:
        st.warning("This image has very little contrast. It may be blank or unlike the training data.")
    if mean_confidence < 60 or low_confidence_share > 35:
        st.warning("Many regions have low model confidence. Treat this result as a rough review only.")
    elif mean_entropy > 0.55:
        st.info("The model is uncertain across the image. Review the overlay rather than relying on the summary alone.")
    cols = st.columns(3)
    for col, label, value in zip(
        cols,
        ["Average level", "Clear regions", "Weak regions"],
        [f"{mean_level:.2f} / 5", f"{clear_share:.1f}%", f"{weak_share:.1f}%"],
    ):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )
    confidence_cols = st.columns(3)
    for col, label, value in zip(
            confidence_cols,
            ["Model confidence", "Low-confidence regions", "Uncertainty"],
            [f"{mean_confidence:.1f}%", f"{low_confidence_share:.1f}%", f"{mean_entropy:.2f}"],
    ):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
                    unsafe_allow_html=True,
                )
    st.caption("Confidence is the model's top-class probability, not a calibrated probability of correctness.")
    st.markdown("#### Scale")
    legend = np.array(
        [[[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in QUALITY_COLORS]],
        dtype=np.uint8,
    )
    st.image(
        np.repeat(legend, 28, axis=0),
        caption="  ".join(f"{i} {QUALITY_NAMES[i]}" for i in range(6)),
        channels="RGB",
        width=520,
    )
    st.caption(f"Input: {original.shape[1]} × {original.shape[0]} px · device: {device}")

with st.expander("View the prepared grayscale input"):
    st.image(display_original, caption="Letterboxed input sent to the model.", use_container_width=True)

if expert_quality is not None:
    with st.expander("View the embedded expert quality map"):
        st.image(
            expert_quality,
            caption="Quality map embedded in the NIST LFFS file. This is reference annotation, not a model prediction.",
            clamp=True,
            use_container_width=True,
        )
        st.caption(f"Source: {source_type} · annotation grid: {expert_quality.shape[1]} × {expert_quality.shape[0]} cells")
