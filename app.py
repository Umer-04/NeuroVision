import os
import time
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

st.set_page_config(
    page_title="CatDogAI | NeuralForge",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- SPLASH SCREEN ----------------
if not st.session_state.get("splash_done", False):
    splash = st.empty()

    with splash.container():
        st.markdown("""
        <style>
        .splash-screen {
            min-height: 65vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 25px 12px;
            background: linear-gradient(145deg, #0b1220, #102b3a);
            border: 1px solid #263852;
            border-radius: 24px;
        }

        .splash-icon {
            font-size: 60px;
            margin-bottom: 18px;
        }

        .splash-title {
            font-size: 42px;
            font-weight: 800;
            color: #f1f5f9;
        }

        .splash-subtitle {
            color: #b7c6d9;
            font-size: 15px;
            margin-top: 10px;
        }

        .splash-brand {
            color: #67e8c2;
            font-size: 21px;
            font-weight: 700;
            margin-top: 42px;
        }

        .splash-tagline {
            color: #8295ac;
            font-size: 13px;
            margin-top: 10px;
        }

        .loading-text {
            color: #b7c6d9;
            font-size: 13px;
            margin-top: 20px;
        }

        .loading-track {
            width: 220px;
            height: 5px;
            margin: 12px auto 0;
            background: #263852;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }

        .loading-fill {
            position: absolute;
            width: 40%;
            height: 100%;
            left: -40%;
            background: #67e8c2;
            border-radius: 10px;
            animation: loadingMove 1.2s linear infinite;
        }

        @keyframes loadingMove {
            from { left: -40%; }
            to   { left: 100%; }
        }
        </style>

        <div class="splash-screen">
            <div class="splash-icon">🧠</div>
            <div class="splash-title">CatDogAI</div>
            <div class="splash-subtitle">
                AI-Powered Image Classification
            </div>
            <div class="splash-brand">
                A NeuralForge Project
            </div>
            <div class="splash-tagline">
                Intelligent Vision. Smarter Results.
            </div>
            <div class="loading-text">
                Initializing AI experience...
            </div>
            <div class="loading-track">
                <div class="loading-fill"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    time.sleep(3)
    st.session_state["splash_done"] = True
    splash.empty()


# ---------------- MAIN APP DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0b1220, #111c30);
    color: #f1f5f9;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    color: #f1f5f9;
}

.hero {
    text-align: center;
    padding: 28px 16px;
    border: 1px solid #263852;
    border-radius: 24px;
    background: linear-gradient(135deg, #16243a, #102b3a);
    margin-bottom: 24px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #b7c6d9;
    font-size: 16px;
}

.section-card {
    background: #172338;
    border: 1px solid #2b3b54;
    border-radius: 18px;
    padding: 22px;
    margin: 16px 0;
}

.result-card {
    background: #142d36;
    border: 1px solid #2c6970;
    border-radius: 20px;
    padding: 24px;
    text-align: center;
    margin-top: 20px;
}

.result-label {
    color: #9fb6c9;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.result-value {
    color: #67e8c2;
    font-size: 34px;
    font-weight: 800;
    margin: 8px 0;
}

.confidence {
    color: #d5e5ef;
    font-size: 17px;
}

div.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    min-height: 48px;
    background: linear-gradient(90deg, #14b8a6, #0d9488);
    color: white;
    font-weight: 700;
    font-size: 16px;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #0d9488, #0f766e);
    color: white;
}

[data-testid="stFileUploader"] {
    background: #121e30;
    border: 1px dashed #4b6685;
    border-radius: 16px;
    padding: 12px;
}

[data-testid="stImage"] img {
    border-radius: 16px;
}

.footer {
    text-align: center;
    color: #8295ac;
    font-size: 13px;
    padding-top: 30px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div style="font-size:48px;">🐱 🐶</div>
    <div class="hero-title">CatDogAI</div>
    <div class="hero-subtitle">
        AI-Powered Image Classification
    </div>
    <div style="color:#67e8c2;font-weight:700;margin-top:12px;">
        Powered by NeuralForge
    </div>
</div>
""", unsafe_allow_html=True)

st.write(
    "Upload an image and let AI determine whether "
    "it contains a cat, a dog, or something else."
)


# ---------------- MODEL SETTINGS ----------------
MODEL_PATH = "models/cat_dog_other_model_v2.pth"
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)

    state = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------- IMAGE UPLOAD ----------------
st.markdown("### 📤 Upload Your Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, PNG, or WEBP image",
    type=["jpg", "jpeg", "png", "webp"],
    help="For best results, use a clear image."
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Your uploaded image",
            use_container_width=True
        )

        if st.button("✨ Analyze Image", type="primary"):
            with st.spinner("AI is analyzing your image..."):
                model = load_model()
                image_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]

                predicted_index = int(
                    probabilities.argmax().item()
                )
                confidence = float(
                    probabilities[predicted_index].item()
                )

            classes = ["Cat", "Dog", "Other"]
            predicted_class = classes[predicted_index]

            icons = {
                "Cat": "🐱",
                "Dog": "🐶",
                "Other": "🌎"
            }

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Prediction Result</div>
                <div style="font-size:48px;margin-top:10px;">
                    {icons[predicted_class]}
                </div>
                <div class="result-value">{predicted_class}</div>
                <div class="confidence">
                    Model score: {confidence * 100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(confidence)

            if predicted_class == "Other":
                st.info(
                    "This image was classified as neither "
                    "a cat nor a dog by the model."
                )

            st.caption(
                "The score is the model's predicted probability, "
                "not a guarantee of correctness."
            )

    except Exception as error:
        st.error("Unable to analyze this image.")
        st.caption(str(error))

else:
    st.markdown("""
    <div class="section-card">
        <h3>🔍 How it works</h3>
        <p>1. Upload an image.</p>
        <p>2. Click Analyze Image.</p>
        <p>3. View the AI prediction and model score.</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    CatDogAI · A NeuralForge Project
</div>
""", unsafe_allow_html=True)