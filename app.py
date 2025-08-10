import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io

st.set_page_config(page_title="Image Processor", layout="wide")

# ===== Feature Functions =====
def remove_background(pil_img):
    """Remove background using rembg."""
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    output_bytes = remove(img_bytes)
    return Image.open(io.BytesIO(output_bytes)).convert("RGBA")

def enhance_image(pil_img):
    """Enhance features (CLAHE + Sharpen)."""
    img_np = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE for L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge([cl, a, b])

    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Sharpen
    enhanced_rgb_float = enhanced_rgb.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(enhanced_rgb_float, (0, 0), 3)
    sharpened = enhanced_rgb_float + (enhanced_rgb_float - blurred) * 1.0
    sharpened = np.clip(sharpened, 0, 1)
    sharpened_rgb = (sharpened * 255).astype(np.uint8)

    return Image.fromarray(sharpened_rgb)

def add_background_color(img_rgba, color_hex):
    """Add solid background color to RGBA image."""
    bg_color = Image.new("RGBA", img_rgba.size, color_hex)
    return Image.alpha_composite(bg_color, img_rgba)

def resize_image(pil_img, size):
    """Resize image to given size."""
    return pil_img.resize(size, Image.LANCZOS)

# ===== Streamlit UI =====
st.title("🖼️ Image Background Remover & Enhancer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📤 Uploaded Image", use_container_width=True)

    # Sidebar options
    st.sidebar.header("⚙️ Operations")
    do_bg_remove = st.sidebar.checkbox("Remove Background")
    do_enhance = st.sidebar.checkbox("Enhance Image Features")

    # Color picker for background
    bg_color = None
    if do_bg_remove:
        bg_color = st.sidebar.color_picker("Pick Background Color", "#FFFFFF")

    # Resize options
    size_option = st.sidebar.selectbox(
        "Resize Image To",
        ["Original", "800x600", "1024x768", "1920x1080"]
    )

    if st.sidebar.button("🚀 Process Image"):
        processed_img = image

        # Background removal
        if do_bg_remove:
            processed_img = remove_background(processed_img)
            if bg_color:
                processed_img = add_background_color(processed_img, bg_color)

        # Enhancement
        if do_enhance:
            processed_img = enhance_image(processed_img)

        # Resize
        if size_option != "Original":
            w, h = map(int, size_option.split("x"))
            processed_img = resize_image(processed_img, (w, h))

        # Show processed image
        st.subheader("✅ Processed Image")
        st.image(processed_img, use_container_width=True)

        # Download option
        buf = io.BytesIO()
        processed_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="💾 Download Image",
            data=byte_im,
            file_name="processed.png",
            mime="image/png"
        )
