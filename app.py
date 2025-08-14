import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import io
from rembg import remove

st.set_page_config(page_title="🎨 Image Processing Tool", layout="wide")
st.title("🖼️ Advanced Image Processing Tool")

# ================== Globals & Quick Colors ==================
quick_colors = {
    "White": (255,255,255),
    "Black": (0,0,0),
    "Blue": (0,0,255),
    "Green": (0,255,0)
}

def hex_color(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

# ================== Image Processing Functions ==================
def remove_background(image_bgr):
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format='PNG')
    output_bytes = remove(img_bytes.getvalue())  # rembg
    return Image.open(io.BytesIO(output_bytes)).convert("RGBA")

def add_background(image_rgba, color=(255, 255, 255)):
    bg = Image.new("RGBA", image_rgba.size, color + (255,))
    return Image.alpha_composite(bg, image_rgba)

def _enhance_max(pil_img):
    has_alpha = (pil_img.mode == "RGBA")
    if has_alpha:
        arr = np.array(pil_img)
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
    else:
        alpha = None
        rgb = np.array(pil_img.convert("RGB"))

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.6, sigmaY=1.6)
    sharp = cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)

    pil_tmp = Image.fromarray(sharp)
    pil_tmp = ImageEnhance.Contrast(pil_tmp).enhance(1.15)
    pil_tmp = ImageEnhance.Color(pil_tmp).enhance(1.20)

    out_rgb = np.array(pil_tmp)
    if alpha is not None:
        out = np.dstack((out_rgb, alpha))
        return Image.fromarray(out, mode="RGBA")
    return pil_tmp

def enhance_image_blend(pil_img, amount):
    amount = float(max(0.0, min(1.0, amount)))
    if amount == 0.0:
        return pil_img.copy()
    enhanced_max = _enhance_max(pil_img)
    if pil_img.mode == "RGBA":
        orig = np.array(pil_img)
        enh  = np.array(enhanced_max)
        alpha = orig[:, :, 3:4]
        blended_rgb = (orig[:, :, :3].astype(np.float32) * (1.0 - amount) +
                       enh[:, :, :3].astype(np.float32) * amount)
        blended_rgb = np.clip(blended_rgb, 0, 255).astype(np.uint8)
        out = np.concatenate([blended_rgb, alpha], axis=2)
        return Image.fromarray(out, mode="RGBA")
    else:
        orig = np.array(pil_img.convert("RGB"))
        enh  = np.array(enhanced_max.convert("RGB"))
        blended_rgb = (orig.astype(np.float32) * (1.0 - amount) +
                       enh.astype(np.float32)  * amount)
        blended_rgb = np.clip(blended_rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(blended_rgb, mode="RGB")

def expand_bbox(x, y, w, h, img_width, img_height, expand_ratio=0.4):
    # Expand bbox by expand_ratio (e.g., 0.4 means 40% larger in all directions)
    x_exp = int(x - w * expand_ratio / 2)
    y_exp = int(y - h * expand_ratio / 2)
    w_exp = int(w * (1 + expand_ratio))
    h_exp = int(h * (1 + expand_ratio))
    # Clamp to image boundaries
    x_exp = max(0, x_exp)
    y_exp = max(0, y_exp)
    w_exp = min(img_width - x_exp, w_exp)
    h_exp = min(img_height - y_exp, h_exp)
    return x_exp, y_exp, w_exp, h_exp

def enhance_faces_only(pil_img, amount, cascade_path="haarcascade_frontalface_default.xml"):
    img_cv = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        st.error("Could not load Haar Cascade XML file for face detection. Please ensure 'haarcascade_frontalface_default.xml' is in the project folder.")
        return pil_img, []
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=3,   # More sensitive
        minSize=(30, 30)  # Detect smaller faces
    )
    img_out = pil_img.copy()
    img_width, img_height = pil_img.size
    expanded_faces = []
    for (x, y, w, h) in faces:
        x_exp, y_exp, w_exp, h_exp = expand_bbox(x, y, w, h, img_width, img_height, expand_ratio=0.4)
        face_crop = pil_img.crop((x_exp, y_exp, x_exp+w_exp, y_exp+h_exp))
        enhanced_face = enhance_image_blend(face_crop, amount)
        img_out.paste(enhanced_face, (x_exp, y_exp))
        expanded_faces.append((x_exp, y_exp, w_exp, h_exp))
    return img_out, expanded_faces

# ================== Streamlit UI ==================
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("📂 Upload Image", type=["png", "jpg", "jpeg"])
remove_bg = st.sidebar.checkbox("Remove Background")
add_bg = st.sidebar.checkbox("Add Background Color")
resize_img = st.sidebar.checkbox("Resize Image")
enhance_img = st.sidebar.checkbox("Enhance Image")
face_only = st.sidebar.checkbox("Enhance Only Face Region")

st.sidebar.markdown("**Background Color**")
color_option = st.sidebar.selectbox("Quick Colors", list(quick_colors.keys()))
selected_color = quick_colors[color_option]
custom_color = st.sidebar.color_picker("Or pick custom", hex_color(selected_color))
if custom_color:
    selected_color = tuple(int(custom_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

resize_presets = ["800x600","1024x768","640x480","Custom"]
resize_choice = st.sidebar.selectbox("Resize Preset", resize_presets)
custom_w = st.sidebar.number_input("Custom Width", min_value=1, value=800) if resize_choice == "Custom" else None
custom_h = st.sidebar.number_input("Custom Height", min_value=1, value=600) if resize_choice == "Custom" else None

# Show appropriate slider
enhance_strength = 0
face_enhance_strength = 0
if enhance_img and not face_only:
    enhance_strength = st.sidebar.slider("Enhancement Strength (Whole Image)", 0, 100, 0)
if face_only:
    face_enhance_strength = st.sidebar.slider("Enhancement Strength (Face Only)", 0, 100, 0)

original_pil = None
processed_img = None

if uploaded_file:
    original_pil = Image.open(uploaded_file).convert("RGBA")
    img = original_pil.copy()

    # Background removal
    if remove_bg:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
        img = remove_background(img_cv)

    # Add solid background
    if add_bg:
        img = add_background(img, selected_color)

    # Resize
    if resize_img:
        if resize_choice == "Custom":
            w, h = custom_w, custom_h
        else:
            w, h = map(int, resize_choice.split("x"))
        img = img.resize((w, h), Image.LANCZOS)

    # Enhancement
    if enhance_img and not face_only:
        amt = enhance_strength / 100.0
        img = enhance_image_blend(img, amount=amt)
    elif face_only:
        amt = face_enhance_strength / 100.0
        cascade_path = "haarcascade_frontalface_default.xml"
        img_rgb = img.convert("RGB")
        img_enh, faces = enhance_faces_only(img_rgb, amt, cascade_path)
        img = img_enh.convert("RGBA")

    processed_img = img

    # Show images side by side with fixed width
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_pil, caption="Original Image", width=300)
    with col2:
        st.image(processed_img, caption="Processed Image", width=300)

    # Download button
    buf = io.BytesIO()
    processed_img.save(buf, format="PNG")
    st.download_button(
        label="Download Processed Image",
        data=buf.getvalue(),
        file_name="processed_image.png",
        mime="image/png"
    )
else:
    st.info("Upload an image to get started.")