# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import io
import sys
from typing import Tuple

st.set_page_config(page_title="Advanced Image Processing Tool", layout="centered")

# ----------------- Helpers & cached resources -----------------
@st.cache_resource
def load_face_cascade():
    # Haar cascade (fast, no extra deps)
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@st.cache_resource
def load_rembg_if_available():
    # lazy import rembg (may not be installed in some hosts)
    try:
        from rembg import remove as rembg_remove
        print("rembg available", file=sys.stderr)
        return rembg_remove
    except Exception as e:
        print("rembg not available:", e, file=sys.stderr)
        return None

face_cascade = load_face_cascade()
rembg_remove = load_rembg_if_available()

# ----------------- Image utilities -----------------
def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)

def np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(arr)

def resize_for_preview(pil_img: Image.Image, max_side: int = 640) -> Image.Image:
    w, h = pil_img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        return pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return pil_img.copy()

# ----------------- Background removal (fast GrabCut preview) -----------------
def grabcut_remove(pil_img: Image.Image) -> Image.Image:
    """Fast GrabCut-based alpha output (RGBA). Works fairly well for preview."""
    img = pil_to_np(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1,65), np.float64)
    fgd = np.zeros((1,65), np.float64)

    # rect inset a little
    rx = max(1, int(0.02 * w))
    ry = max(1, int(0.02 * h))
    rw = max(2, w - 2*rx)
    rh = max(2, h - 2*ry)
    rect = (rx, ry, rw, rh)

    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        print("grabCut error:", e, file=sys.stderr)

    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    alpha = (mask2 * 255).astype('uint8')
    rgba = np.dstack((img, alpha))
    return np_to_pil(rgba)

def rembg_remove_highres(pil_img: Image.Image):
    """Use rembg if available; raises if not available."""
    if rembg_remove is None:
        raise RuntimeError("rembg not available")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    out = rembg_remove(buf.getvalue())
    return Image.open(io.BytesIO(out)).convert("RGBA")

# ----------------- Face-only enhancement -----------------
def detect_faces_np(img_rgb: np.ndarray) -> list:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    return faces

def make_face_mask(img_rgb: np.ndarray, faces, expand = 1.2, feather=15) -> np.ndarray:
    """Return single-channel mask [0..255] with soft edges."""
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x,y,fw,fh) in faces:
        cx, cy = x + fw//2, y + fh//2
        rx = int((fw/2) * expand)
        ry = int((fh/2) * expand)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    if feather>0:
        k = max(3, feather//2*2+1)
        mask = cv2.GaussianBlur(mask, (k,k), feather)
    return mask

def enhance_face_region(pil_img: Image.Image, clip_limit: float, sharpness: float, preview: bool=False) -> Image.Image:
    """Apply CLAHE and unsharp only inside face mask. preview=True expects small image."""
    # work on RGB array
    img_rgb = pil_to_np(pil_img.convert("RGB"))
    faces = detect_faces_np(img_rgb)
    if len(faces) == 0:
        # nothing to do
        return pil_img

    mask = make_face_mask(img_rgb, faces, expand=1.25, feather=25 if not preview else 11)
    mask_norm = mask.astype(np.float32)/255.0
    mask_3c = np.dstack([mask_norm]*3)

    # CLAHE on L channel
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    l_enh = clahe.apply(l)
    lab_enh = cv2.merge([l_enh, a, b])
    rgb_enh = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2RGB)

    # unsharp
    rgb_enh_f = rgb_enh.astype(np.float32)/255.0
    blurred = cv2.GaussianBlur(rgb_enh_f, (0,0), 3)
    unsharp = np.clip(rgb_enh_f + (rgb_enh_f - blurred) * sharpness, 0, 1)
    unsharp_uint8 = (unsharp*255).astype(np.uint8)

    # Blend only in mask region
    result = (img_rgb*(1-mask_3c) + unsharp_uint8*mask_3c).astype(np.uint8)
    return np_to_pil(result)

# ----------------- UI: Layout + controls -----------------
st.markdown("<h2 style='text-align:center'>Advanced Image Processing Tool</h2>", unsafe_allow_html=True)

# Main previews area: two columns
orig_col, proc_col = st.columns([1,1])

with orig_col:
    st.markdown("**Original Image**")
    orig_placeholder = st.empty()  # will show original
with proc_col:
    st.markdown("**Processed Image (Preview)**")
    proc_placeholder = st.empty()  # will show processed preview

# Controls below previews (single full-width area)
st.markdown("---")
st.markdown("### Controls")
cols = st.columns([1,1,1,1])

with cols[0]:
    uploaded = st.file_uploader("Upload Image (jpg/png)", type=["jpg","jpeg","png"])
    show_original_size = st.checkbox("Show original size", value=False)
with cols[1]:
    do_bg_remove = st.checkbox("Remove Background (preview uses fast GrabCut)", value=True)
    use_rembg_final = st.checkbox("Use rembg for final (high quality)", value=True)
with cols[2]:
    do_enhance = st.checkbox("Enhance Face Only", value=True)
    clip_limit = st.slider("CLAHE clip limit", min_value=1.0, max_value=5.0, value=2.0, step=0.1) if do_enhance else 2.0
with cols[3]:
    sharp_strength = st.slider("Sharpen strength", min_value=0.0, max_value=2.5, value=0.8, step=0.05) if do_enhance else 0.0

# Additional controls row
cols2 = st.columns([1,1,1,1])
with cols2[0]:
    add_bg_color = st.checkbox("Add Background Color after removal", value=False)
with cols2[1]:
    bg_color_hex = st.color_picker("Background Color", "#FFFFFF") if add_bg_color else "#FFFFFF"
with cols2[2]:
    resize_option = st.selectbox("Preview / Final Size", options=["Preview size", "Original", "800x600", "1024x768"])
with cols2[3]:
    apply_button = st.button("Apply (final, high-res)")
    download_button = None

# Processing pipeline (preview)
preview_max_side = 640  # preview downscale for responsiveness

if uploaded is None:
    orig_placeholder.info("Upload an image to begin")
    proc_placeholder.info("Processed preview will appear here")
else:
    # Load PIL image
    uploaded_pil = Image.open(uploaded)
    if not show_original_size:
        display_orig = resize_for_preview(uploaded_pil, max_side=800)
    else:
        display_orig = uploaded_pil.copy()

    orig_placeholder.image(display_orig, use_column_width=True)

    # Build preview image (low-res, fast)
    # Start from downscaled copy
    preview_input = resize_for_preview(uploaded_pil, max_side=preview_max_side)

    # 1) background removal (preview fast: GrabCut)
    if do_bg_remove:
        try:
            preview_removed = grabcut_remove(preview_input)
        except Exception as e:
            print("grabcut preview error:", e, file=sys.stderr)
            preview_removed = preview_input.copy()
    else:
        preview_removed = preview_input.copy()

    # 2) if add_bg_color and removal done, composite background for preview
    if do_bg_remove and add_bg_color:
        # color hex -> RGBA tuple
        hexc = bg_color_hex.lstrip('#')
        r,g,b = tuple(int(hexc[i:i+2],16) for i in (0,2,4))
        bg = Image.new("RGBA", preview_removed.size, (r,g,b,255))
        if preview_removed.mode != "RGBA":
            preview_removed = preview_removed.convert("RGBA")
        preview_removed = Image.alpha_composite(bg, preview_removed).convert("RGB")
    else:
        preview_removed = preview_removed.convert("RGB")

    # 3) face-only enhancement on preview (fast low-res)
    if do_enhance:
        try:
            preview_processed = enhance_face_region(preview_removed, clip_limit, sharp_strength, preview=True)
        except Exception as e:
            print("preview enhance error:", e, file=sys.stderr)
            preview_processed = preview_removed
    else:
        preview_processed = preview_removed

    # Show preview
    proc_placeholder.image(preview_processed, use_column_width=True)

    # If user clicks Apply (final high-res): run final pipeline (may use rembg)
    if apply_button:
        st.info("Running final high-quality processing — this may take a few seconds.")
        # operate on original uploaded_pil (full resolution)
        final_img = uploaded_pil.convert("RGBA")

        # 1) high-quality background removal if requested
        if do_bg_remove:
            # If rembg available and user wants it, use it; otherwise use GrabCut high-res
            if use_rembg_final and rembg_remove is not None:
                try:
                    final_img = rembg_remove_highres = rembg_remove(io.BytesIO(final_img.tobytes()) if False else None)  # placeholder
                    # We must call rembg_remove with bytes; simpler to use helper:
                except Exception as e:
                    # fallback to helper wrapper below
                    print("rembg direct usage placeholder", file=sys.stderr)
                # use wrapper function defined earlier to safely call rembg
                try:
                    final_img = rembg_remove_highres = rembg_remove_helper(uploaded_pil)
                except Exception as e:
                    print("rembg failed, falling back to GrabCut:", e, file=sys.stderr)
                    final_img = grabcut_remove(uploaded_pil).convert("RGBA")
            else:
                final_img = grabcut_remove(uploaded_pil).convert("RGBA")

        # 2) add bg color if requested
        if do_bg_remove and add_bg_color:
            hexc = bg_color_hex.lstrip('#')
            r,g,b = tuple(int(hexc[i:i+2],16) for i in (0,2,4))
            bg = Image.new("RGBA", final_img.size, (r,g,b,255))
            final_img = Image.alpha_composite(bg, final_img).convert("RGB")
        else:
            # if no background removal but still want bg color? ignore
            if final_img.mode == "RGBA":
                final_img = final_img.convert("RGB")

        # 3) final face enhancement (full-res)
        if do_enhance:
            try:
                final_img = enhance_face_region(final_img, clip_limit, sharp_strength, preview=False)
            except Exception as e:
                print("final enhance error:", e, file=sys.stderr)

        # 4) final resize if requested
        if resize_option != "Preview size" and resize_option != "Original":
            w,h = map(int, resize_option.split("x"))
            final_img = final_img.resize((w,h), Image.LANCZOS)

        # Show final
        st.subheader("Final Result")
        st.image(final_img, use_column_width=True)

        # Download button
        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("Download final PNG", data=byte_im, file_name="processed_final.png", mime="image/png")

# ----------------- End -----------------
