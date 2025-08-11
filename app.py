import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io

st.set_page_config(page_title="Image Processor", layout="centered")

# ===== Feature Functions =====
def remove_background(pil_img):
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    output_bytes = remove(img_bytes)
    return Image.open(io.BytesIO(output_bytes)).convert("RGBA")

def enhance_face_only(pil_img, clip_limit=2.0, sharp_strength=1.0):
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    mask = np.zeros_like(gray)
    for (x, y, w, h) in faces:
        cv2.ellipse(mask, (x + w//2, y + h//2), (w//2, h//2), 0, 0, 360, 255, -1)

    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    blurred = cv2.GaussianBlur(rgb_enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(rgb_enhanced, 1 + sharp_strength, blurred, -sharp_strength, 0)

    # Blend only in face region
    mask_3ch = cv2.merge([mask] * 3)
    final_img = np.where(mask_3ch > 0, sharpened, img_np)

    return Image.fromarray(final_img)

def add_background_color(img_rgba, color_hex):
    bg_color = Image.new("RGBA", img_rgba.size, color_hex)
    return Image.alpha_composite(bg_color, img_rgba)

def resize_image(pil_img, size):
    return pil_img.resize(size, Image.LANCZOS)

# ===== Streamlit UI =====
st.title("🖼️ Image Background Remover & Face Enhancer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📤 Uploaded Image", use_container_width=True)

    st.subheader("⚙️ Operations")
    do_bg_remove = st.checkbox("Remove Background")
    do_enhance = st.checkbox("Enhance Face Only")

    if do_bg_remove:
        bg_color = st.color_picker("Pick Background Color", "#FFFFFF")
    else:
        bg_color = None

    if do_enhance:
        clip_limit = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.1)
        sharp_strength = st.slider("Sharpen Strength", 0.0, 3.0, 1.0, 0.1)
    else:
        clip_limit, sharp_strength = 2.0, 1.0

    size_option = st.selectbox("Resize Image To", ["Original", "800x600", "1024x768", "1920x1080"])

    if st.button("🚀 Process Image"):
        processed_img = image

        if do_bg_remove:
            processed_img = remove_background(processed_img)
            if bg_color:
                processed_img = add_background_color(processed_img, bg_color)

        if do_enhance:
            processed_img = enhance_face_only(processed_img, clip_limit, sharp_strength)

        if size_option != "Original":
            w, h = map(int, size_option.split("x"))
            processed_img = resize_image(processed_img, (w, h))

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original", use_container_width=True)
        col2.image(processed_img, caption="Processed", use_container_width=True)

        buf = io.BytesIO()
        processed_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("💾 Download Image", data=byte_im, file_name="processed.png", mime="image/png")
