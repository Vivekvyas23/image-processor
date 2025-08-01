import streamlit as st
from PIL import Image
import numpy as np
import cv2
from rembg import remove
import io

# --- IMAGE PROCESSING FUNCTIONS ---
def remove_bg(image):
    return Image.open(io.BytesIO(remove(image))).convert("RGBA")

def enhance_features(image):
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    img_np = np.array(image)
    rgb, alpha = img_np[:, :, :3], img_np[:, :, 3]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    enhanced_rgb = cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(enhanced_rgb, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced_rgb, 1.5, blurred, -0.5, 0)
    return Image.fromarray(np.dstack((sharpened, alpha)), "RGBA")

def add_solid_bg(image, color):
    bg = Image.new("RGBA", image.size, color+(255,))
    return Image.alpha_composite(bg, image).convert("RGB")

# --- STREAMLIT UI ---
st.set_page_config(page_title="Image Processing Tool", layout="centered")
st.title("🖼 Image Processing Tool")

uploaded = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Original Image", use_container_width=True)

    st.subheader("Select Operations")
    remove_bg_option = st.checkbox("Remove Background")
    add_bg_option = st.checkbox("Add Solid Background Color")
    enhance_option = st.checkbox("Enhance Features")
    resize_option = st.checkbox("Resize Image")

    bg_color = (255, 255, 255)
    if add_bg_option:
        bg_color_hex = st.color_picker("Pick Background Color", "#FFFFFF")
        bg_color = tuple(int(bg_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    new_size = None
    if resize_option:
        width = st.number_input("Width", value=800)
        height = st.number_input("Height", value=600)
        new_size = (width, height)

    if st.button("🚀 Process Image"):
        result = img
        if remove_bg_option:
            result = remove_bg(result)
            if add_bg_option:
                result = add_solid_bg(result, bg_color)
        if enhance_option:
            result = enhance_features(result)
        if resize_option and new_size:
            result = result.resize(new_size, Image.LANCZOS)

        st.image(result, caption="Processed Image", use_container_width=True)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button("⬇ Download Image", buf.getvalue(), "processed.png", "image/png")