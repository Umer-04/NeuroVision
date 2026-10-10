
import time
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms, models


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NeuroVision | AI Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cat_dog_other_model_v2.pth"
DEVICE = torch.device("cpu")

CLASSES = ["Cat", "Dog", "Other"]

CLASS_INFO = {
    "Cat": {"icon": "🐱", "description": "A cat was detected."},
    "Dog": {"icon": "🐶", "description": "A dog was detected."},
    "Other": {
        "icon": "🌎",
        "description": "The image was classified as something other than a cat or dog.",
    },
}

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------- DESIGN SYSTEM ----------------
st.markdown("""
<style>
:root {
    --bg: #080d18;
    --panel: #111b2c;
    --border: #25354c;
    --muted: #9aabc2;
    --text: #edf4ff;
    --accent: #55e6c1;
}

.stApp {
    background:
        radial-gradient(ellipse at 50% -15%, #183d4b 0%, transparent 48%),
        linear-gradient(180deg, #080d18 0%, #0b1423 100%);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1120px;
    padding: 2rem 1.5rem 3rem;
}

h1, h2, h3, p, label {
    color: var(--text);
}

p, li {
    line-height: 1.65;
}

.hero {
    position: relative;
    overflow: hidden;
    text-align: center;
    padding: 45px 22px;
    margin-bottom: 26px;
    border: 1px solid #2a4859;
    border-radius: 26px;
    background: linear-gradient(135deg, #14253a, #102b36 65%, #102036);
    box-shadow: 0 18px 55px #00000030;
}

.hero-icon {
    font-size: 48px;
    margin-bottom: 10px;
}

.hero-title {
    font-size: clamp(34px, 5vw, 54px);
    font-weight: 850;
    letter-spacing: -1.8px;
    line-height: 1.15;
    margin-bottom: 12px;
}

.hero-title span {
    color: var(--accent);
}

.hero-subtitle {
    color: #c2d0e1;
    font-size: 16px;
    max-width: 590px;
    margin: 0 auto;
}

.brand-pill {
    display: inline-block;
    margin-top: 22px;
    padding: 7px 14px;
    border: 1px solid #38675f;
    border-radius: 30px;
    background: #14332f;
    color: #8ff4d8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.section-heading {
    font-size: 23px;
    font-weight: 750;
    margin: 24px 0 6px;
}

.section-description {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 16px;
}

.panel {
    background: linear-gradient(145deg, #121f32, #101a2a);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 18px;
}

.feature-card {
    height: 100%;
    min-height: 125px;
    background: #111d2e;
    border: 1px solid #26374d;
    border-radius: 17px;
    padding: 18px;
}

.feature-icon {
    font-size: 27px;
    margin-bottom: 8px;
}

.feature-title {
    font-weight: 750;
    margin-bottom: 5px;
}

.feature-description {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
}

.result-card {
    background: linear-gradient(145deg, #12312f, #11252f);
    border: 1px solid #2c7567;
    border-radius: 23px;
    text-align: center;
    padding: 27px 20px;
    margin: 16px 0 18px;
}

.result-eyebrow {
    color: #a8c9c9;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.result-icon {
    font-size: 58px;
    margin: 13px 0 5px;
}

.result-name {
    color: var(--accent);
    font-size: clamp(30px, 5vw, 42px);
    font-weight: 850;
    margin-bottom: 8px;
}

.result-description {
    color: #c6d9df;
    font-size: 14px;
}

.metric-card {
    background: #121f32;
    border: 1px solid #283a52;
    border-radius: 16px;
    padding: 17px;
    margin-bottom: 10px;
}

.metric-label {
    color: #a9bad0;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: #f1f7ff;
    font-size: 25px;
    font-weight: 800;
    margin-top: 5px;
}

div.stButton > button {
    width: 100%;
    min-height: 49px;
    border: 1px solid #56e5c0;
    border-radius: 13px;
    background: linear-gradient(100deg, #17b99d, #0d8d82);
    color: #061b1a;
    font-weight: 800;
    font-size: 15px;
    transition: all .2s ease;
}

div.stButton > button:hover {
    border-color: #91ffe1;
    background: linear-gradient(100deg, #51e6c1, #23bba4);
    color: #061b1a;
    transform: translateY(-1px);
}

div.stButton > button:focus {
    box-shadow: 0 0 0 3px #55e6c140;
}

[data-testid="stFileUploader"] {
    background: #0c1727;
    border: 1px dashed #48647f;
    border-radius: 16px;
    padding: 14px;
}

[data-testid="stFileUploaderDropzone"] {
    background: #0d192a;
    border: 0;
}

[data-testid="stImage"] img {
    border-radius: 17px;
}

[data-testid="stProgressBar"] > div {
    background: #24364a;
    border-radius: 20px;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #14b8a6, #70f4c9);
}

div[data-testid="stAlert"] {
    border-radius: 13px;
}

.small-note {
    color: #8fa3bb;
    font-size: 12px;
    line-height: 1.6;
}

.footer {
    border-top: 1px solid #243247;
    text-align: center;
    color: #8497af;
    font-size: 12px;
    margin-top: 38px;
    padding-top: 22px;
}

@media (max-width: 640px) {
    .block-container {
        padding: 1rem .8rem 2rem;
    }

    .hero {
        padding: 30px 14px;
        border-radius: 20px;
    }

    .hero-icon {
        font-size: 38px;
    }

    .hero-title {
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 14px;
    }

    .panel {
        padding: 15px;
        border-radius: 16px;
    }

    .result-card {
        padding: 22px 14px;
    }

    .section-heading {
        font-size: 21px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- SPLASH SCREEN ----------------
if not st.session_state.get("splash_done", False):
    splash = st.empty()

    with splash.container():
        st.markdown("""
        <div class="hero" style="margin-top:45px; padding:55px 18px;">
            <div class="hero-icon">🧠</div>
            <div class="hero-title">Neuro<span>Vision</span></div>
            <div class="hero-subtitle">
                Intelligent image recognition, powered by AI.
            </div>
            <div class="brand-pill">A NEURALFORGE PROJECT</div>
            <p style="color:#9aabc2; margin-top:25px;">
                Preparing your AI experience...
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        .splash-track {
            height: 4px;
            max-width: 300px;
            margin: 0 auto;
            border-radius: 10px;
            overflow: hidden;
            background: #25354c;
        }
        .splash-fill {
            width: 40%;
            height: 100%;
            background: #55e6c1;
            animation: splash-slide 1.2s linear infinite;
        }
        @keyframes splash-slide {
            from { transform: translateX(-110%); }
            to { transform: translateX(260%); }
        }
        </style>
        <div class="splash-track">
            <div class="splash-fill"></div>
        </div>
        """, unsafe_allow_html=True)

    time.sleep(1.5)
    st.session_state["splash_done"] = True
    splash.empty()


# ---------------- MODEL LOADER ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    state = torch.load(
        str(MODEL_PATH),
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="hero-icon">🔬</div>
    <div class="hero-title">Neuro<span>Vision</span></div>
    <div class="hero-subtitle">
        An intelligent image classifier that predicts whether
        your image contains a cat, a dog, or another subject.
    </div>
    <div class="brand-pill">POWERED BY NEURALFORGE</div>
</div>
""", unsafe_allow_html=True)


# ---------------- FEATURES ----------------
feature_columns = st.columns(3)

features = [
    ("🧠", "AI-Powered", "Predictions from your trained vision model."),
    ("⚡", "Fast Analysis", "Analyze an image in just a few moments."),
    ("📱", "Responsive UI", "Designed for desktop and mobile screens."),
]

for column, (icon, title, description) in zip(feature_columns, features):
    with column:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-description">{description}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- MAIN CONTENT ----------------
st.markdown('<div class="section-heading">Analyze an Image</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">'
    'Upload a clear image to get your model prediction and class scores.'
    '</div>',
    unsafe_allow_html=True,
)

left_column, right_column = st.columns([1, 1], gap="large")

with left_column:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 📤 Upload image")

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, PNG, or WEBP file",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear image for best results.",
        key="image_uploader",
    )

    st.markdown(
        '<p class="small-note">Supported formats: JPG, JPEG, PNG and WEBP.</p>',
        unsafe_allow_html=True,
    )

    analyze_clicked = st.button(
        "✨ Analyze Image",
        type="primary",
        disabled=uploaded_file is None,
    )

    st.markdown('</div>', unsafe_allow_html=True)

with right_column:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 💡 How it works")
    st.markdown("""
    1. Upload an image from your device.
    2. Select **Analyze Image**.
    3. Review the predicted class and probability scores.
    """)
    st.markdown(
        '<p class="small-note">'
        'The model predicts one of three classes: Cat, Dog, or Other.'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- IMAGE PREVIEW ----------------
image = None

if uploaded_file is not None:
    try:
        image = ImageOps.exif_transpose(
            Image.open(uploaded_file)
        ).convert("RGB")

        st.markdown('<div class="section-heading">Your Image</div>',
                    unsafe_allow_html=True)
        st.image(
            image,
            caption="Uploaded image preview",
            use_container_width=True,
        )

    except Exception:
        st.error("This image could not be opened. Please try another file.")


# ---------------- PREDICTION ----------------
if analyze_clicked and image is not None:
    try:
        with st.spinner("Loading AI model and analyzing image..."):
            model = load_model()
            image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]

            predicted_index = int(probabilities.argmax().item())
            predicted_class = CLASSES[predicted_index]
            confidence = float(probabilities[predicted_index].item())

            scores = {
                name: float(probabilities[index].item())
                for index, name in enumerate(CLASSES)
            }

        st.markdown('<div class="section-heading">Prediction Results</div>',
                    unsafe_allow_html=True)

        result = CLASS_INFO[predicted_class]

        st.markdown(f"""
        <div class="result-card">
            <div class="result-eyebrow">Predicted Class</div>
            <div class="result-icon">{result["icon"]}</div>
            <div class="result-name">{predicted_class}</div>
            <div class="result-description">{result["description"]}</div>
        </div>
        """, unsafe_allow_html=True)

        metric_columns = st.columns(3)

        for column, class_name in zip(metric_columns, CLASSES):
            with column:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{class_name}</div>
                    <div class="metric-value">{scores[class_name] * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### Class probability scores")

        for class_name in CLASSES:
            score = scores[class_name]
            st.markdown(
                f"**{CLASS_INFO[class_name]['icon']} {class_name}** — "
                f"{score * 100:.2f}%"
            )
            st.progress(score)

        st.info(
            "These percentages are the model's softmax scores. "
            "They are not guaranteed to represent real-world accuracy "
            "or calibrated confidence."
        )

    except FileNotFoundError as error:
        st.error("The AI model file could not be found.")
        st.caption(str(error))
        st.info(
            "Check that the model exists at "
            "`models/cat_dog_other_model_v2.pth`."
        )

    except Exception as error:
        st.error("Unable to analyze this image.")
        st.caption(f"{type(error).__name__}: {error}")


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <strong>NeuroVision</strong> · A NeuralForge Project
    <br>
    AI-powered image classification
</div>
""", unsafe_allow_html=True)
