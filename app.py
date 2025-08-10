import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import sys

# ---------- FUNCTIONS ----------
@st.cache_resource
def load_rembg():
    print("Loading rembg model...", file=sys.stderr)
    return remove  # caching function so it’s not reloaded every time

def enhance_features(image):
    rgb = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge([cl, a, b])

    enhanced_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    enhanced_rgb_float = enhanced_rgb.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(enhanced_rgb_float, (0, 0), 3)
    unsharp_mask = enhanced_rgb_float - blurred
    sharpened_rgb_float = enhanced_rgb_float + unsharp_mask * 1.0
    sharpened_rgb_float = np.clip(sharpened_rgb_float, 0, 1)
    sharpened_rgb = (sharpened_rgb_float * 255).astype(np.uint8)

    return Image.fromarray(sharpened_rgb)

def add_solid_background(image, color):
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    bg = Image.new("RGBA", image.size, color)
    return Image.alpha_composite(bg, image)

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Image Processor", layout="wide")
st.title("🖼 Image Background Remover & Enhancer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGBA")
    st.image(image, caption="📷 Original Image", use_column_width=True)

    st.write("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        remove_bg = st.checkbox("Remove Background", value=True)
        enhance = st.checkbox("Enhance Features", value=False)

    with col2:
        bg_color = st.color_picker("Background Color", "#FFFFFF")
        apply_bg = st.checkbox("Apply Solid Background", value=False)

    with col3:
        download_ready = False
        process_btn = st.button("🚀 Process Image")

    if process_btn:
        try:
            processed_img = image
            if remove_bg:
                st.write("Removing background...")
                rembg_func = load_rembg()
                processed_img = Image.open(io.BytesIO(rembg_func(np.array(processed_img))))

            if enhance:
                st.write("Enhancing image...")
                processed_img = enhance_features(processed_img)

            if apply_bg:
                st.write("Adding solid background...")
                processed_img = add_solid_background(processed_img, bg_color)

            st.image(processed_img, caption="✅ Processed Image", use_column_width=True)

            buf = io.BytesIO()
            processed_img.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="📥 Download Image",
                data=byte_im,
                file_name="processed_image.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
            print(f"Error: {e}", file=sys.stderr)
