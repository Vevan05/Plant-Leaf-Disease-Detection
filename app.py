import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# STYLES
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Theme-aware tokens: pull from Streamlit's own CSS vars so this works in
       both light and dark mode, instead of hardcoding colors that only work
       on a light background. */
    :root {
        --card-bg: var(--secondary-background-color, #f7faf7);
        --card-border: rgba(128, 128, 128, 0.25);
        --muted-text: var(--text-color, #374151);
    }

    .hero {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #66bb6a 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0 0 0.4rem 0; font-size: 2.1rem; color: white; }
    .hero p { margin: 0; opacity: 0.92; font-size: 1.02rem; color: white; }

    .metric-card {
        background: var(--card-bg);
        color: var(--muted-text);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        border: 1px solid var(--card-border);
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-card b { font-size: 1.3rem; }
    .metric-card span { opacity: 0.7; }

    .result-card {
        background: var(--card-bg);
        color: var(--muted-text);
        border-radius: 16px;
        padding: 1.5rem 1.7rem;
        border: 1px solid var(--card-border);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .result-card h2, .result-card p { color: var(--muted-text); }

    .healthy-badge {
        background: #e6f4ea;
        color: #1b5e20;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .disease-badge {
        background: #fdecea;
        color: #b3261e;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .footer-note {
        text-align: center;
        opacity: 0.65;
        font-size: 0.82rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid var(--card-border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# CLASS METADATA
# Maps raw model labels -> readable crop / condition / healthy flag / tips
# ----------------------------------------------------------------------------
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

GENERAL_TIPS = {
    "fungal": "Improve air circulation, avoid overhead watering, and remove infected leaves. A copper- or sulfur-based fungicide is commonly used, but confirm the right product for your crop with a local agronomist.",
    "bacterial": "Remove and destroy infected plant material, avoid working with wet plants, and rotate crops. Copper-based bactericides can help; consult local guidance for dosage.",
    "viral": "There is no cure once infected — focus on controlling the insect vectors (e.g. whiteflies, aphids) and removing infected plants to stop spread.",
    "pest": "Inspect the underside of leaves, introduce natural predators where possible, and consider miticide/insecticide treatment if the infestation is severe.",
    "healthy": "No action needed. Keep up good watering, spacing, and nutrient practices to maintain plant health.",
}


def parse_class_name(raw: str):
    """Split a raw PlantVillage label into (crop, condition, is_healthy, category)."""
    crop, _, condition = raw.partition("___")
    crop = crop.replace("_", " ").replace(",", ",").strip()
    is_healthy = condition.lower() == "healthy"
    condition_display = condition.replace("_", " ").strip()

    cond_lower = condition.lower()
    if is_healthy:
        category = "healthy"
    elif "virus" in cond_lower or "curl" in cond_lower or "mosaic" in cond_lower or "greening" in cond_lower:
        category = "viral"
    elif "bacterial" in cond_lower:
        category = "bacterial"
    elif "mite" in cond_lower:
        category = "pest"
    else:
        category = "fungal"

    return crop, condition_display, is_healthy, category


# ----------------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_leaf_disease_model.keras")


model_load_error = None
try:
    model = load_model()
except Exception as e:
    model = None
    model_load_error = str(e)

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 About this Project")
    st.write(
        "A deep learning system that identifies plant diseases from leaf images, "
        "built with **ResNet50 transfer learning** on the **PlantVillage** dataset."
    )

    st.markdown("### 🧠 Model Details")
    st.markdown(
        """
        - **Base model:** ResNet50 (ImageNet weights)
        - **Input size:** 224 × 224 × 3
        - **Classes:** 38 crop/disease categories
        - **Head:** GAP → Dense(512, ReLU) → Dropout(0.3) → Dense(38, Softmax)
        - **Training:** Feature extraction + fine-tuning (last 50 layers)
        """
    )

    st.markdown("### 🌱 Supported Crops")
    crops = sorted({parse_class_name(c)[0] for c in CLASS_NAMES})
    st.write(", ".join(crops))

    st.markdown("### ⚙️ Settings")
    show_top5 = st.checkbox("Show top-5 predictions", value=True)
    confidence_threshold = st.slider(
        "Low-confidence warning threshold (%)", 0, 100, 60
    )

    st.markdown("---")
    st.caption("Built with TensorFlow, Keras & Streamlit · For educational use")

# ----------------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🌿 Plant Leaf Disease Detection</h1>
        <p>Upload a leaf photo and get an instant diagnosis powered by a ResNet50 deep learning model
        trained on 38 crop and disease categories.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top stat strip
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        '<div class="metric-card"><b>38</b><br><span style="font-size:0.85rem;">Disease classes</span></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="metric-card"><b>{len(crops)}</b><br><span style="font-size:0.85rem;">Crop species</span></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="metric-card"><b>ResNet50</b><br><span style="font-size:0.85rem;">Backbone architecture</span></div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        '<div class="metric-card"><b>224×224</b><br><span style="font-size:0.85rem;">Input resolution</span></div>',
        unsafe_allow_html=True,
    )

st.write("")

if model_load_error:
    st.error(
        "⚠️ Could not load the model file `final_leaf_disease_model.keras`. "
        "Make sure it exists in the project root (see the README's training/installation steps)."
    )
    with st.expander("Error details"):
        st.code(model_load_error)

# ----------------------------------------------------------------------------
# MAIN TABS
# ----------------------------------------------------------------------------
tab_predict, tab_classes, tab_about = st.tabs(
    ["🔍 Diagnose a Leaf", "📋 Class Reference", "ℹ️ How it Works"]
)

# ---------------- TAB 1: PREDICTION ----------------
with tab_predict:
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.subheader("Upload an image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image (JPG, JPEG, or PNG)",
            type=["jpg", "jpeg", "png"],
        )
        st.caption(
            "Tip: use a clear, well-lit photo of a single leaf against a plain background "
            "for the most reliable prediction."
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with right:
        st.subheader("Diagnosis")

        if uploaded_file is None:
            st.info("👈 Upload a leaf image to see the prediction here.")
        elif model is None:
            st.warning("Model is not loaded, so a prediction can't be generated right now.")
        else:
            with st.spinner("Analyzing leaf image..."):
                img = image.resize((224, 224))
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                start = time.time()
                prediction = model.predict(img_array, verbose=0)
                elapsed = time.time() - start

            predicted_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            raw_label = CLASS_NAMES[predicted_index]
            crop, condition, is_healthy, category = parse_class_name(raw_label)

            badge_html = (
                '<span class="healthy-badge">✅ Healthy</span>'
                if is_healthy
                else '<span class="disease-badge">⚠️ Disease Detected</span>'
            )

            st.markdown(
                f"""
                <div class="result-card">
                    {badge_html}
                    <h2 style="margin:0.6rem 0 0.1rem 0;">{crop}</h2>
                    <p style="margin:0;font-size:1.05rem;opacity:0.85;">{condition}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Confidence", f"{confidence * 100:.2f}%")
            with m2:
                st.metric("Inference time", f"{elapsed * 1000:.0f} ms")

            st.progress(min(confidence, 1.0))

            if confidence * 100 < confidence_threshold:
                st.warning(
                    f"Confidence is below your {confidence_threshold}% threshold. "
                    "Consider uploading a clearer or closer image of the leaf."
                )

            st.markdown("##### 🩺 Suggested next step")
            st.write(GENERAL_TIPS.get(category, GENERAL_TIPS["healthy"]))
            if not is_healthy:
                st.caption(
                    "This is general guidance only, not a substitute for advice from a "
                    "local agricultural extension office or plant pathologist."
                )

            if show_top5:
                st.markdown("##### 📊 Top-5 Predictions")
                probs = prediction[0]
                top5_idx = np.argsort(probs)[-5:][::-1]
                rows = []
                for idx in top5_idx:
                    c, cond, _, _ = parse_class_name(CLASS_NAMES[idx])
                    rows.append(
                        {"Crop": c, "Condition": cond, "Probability": float(probs[idx])}
                    )
                df = pd.DataFrame(rows)
                st.bar_chart(df.set_index("Crop")["Probability"])
                st.dataframe(
                    df.style.format({"Probability": "{:.2%}"}),
                    use_container_width=True,
                    hide_index=True,
                )

# ---------------- TAB 2: CLASS REFERENCE ----------------
with tab_classes:
    st.subheader("All 38 supported classes")
    st.write(
        "Every category the model was trained to recognize, grouped by crop. "
        "Healthy leaves are included as a baseline class for each crop where available."
    )

    ref_rows = []
    for raw in CLASS_NAMES:
        crop, condition, is_healthy, category = parse_class_name(raw)
        ref_rows.append(
            {
                "Crop": crop,
                "Condition": condition,
                "Status": "Healthy" if is_healthy else "Disease",
                "Category": category.capitalize(),
            }
        )
    ref_df = pd.DataFrame(ref_rows).sort_values(["Crop", "Status"])
    st.dataframe(ref_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download class list as CSV",
        ref_df.to_csv(index=False).encode("utf-8"),
        file_name="leaf_disease_classes.csv",
        mime="text/csv",
    )

# ---------------- TAB 3: ABOUT / HOW IT WORKS ----------------
with tab_about:
    st.subheader("How the model works")

    st.markdown(
        """
        **1. Transfer learning backbone**
        The model starts from **ResNet50** pretrained on ImageNet, with the original
        classification head removed (`include_top=False`), so it can reuse general-purpose
        visual features learned from millions of images.

        **2. Custom classification head**
        ```text
        ResNet50
            ↓
        GlobalAveragePooling2D
            ↓
        Dense (512, ReLU)
            ↓
        Dropout (0.3)
            ↓
        Dense (38, Softmax)
        ```

        **3. Two-phase training**
        - **Phase 1 — Feature extraction:** ResNet50 layers are frozen; only the new head is
          trained, at a learning rate of `0.001`.
        - **Phase 2 — Fine-tuning:** the last 50 ResNet50 layers are unfrozen and trained
          at a much lower learning rate of `0.00001`, so the backbone adapts to leaf imagery
          without losing its pretrained knowledge.

        **4. Regularization & callbacks**
        - **Data augmentation:** rotation, zoom, shear, and horizontal flips improve generalization.
        - **EarlyStopping:** stops training when validation loss stalls (`patience=5`), restoring the best weights.
        - **ReduceLROnPlateau:** shrinks the learning rate when progress stalls.
        - **ModelCheckpoint:** saves only the best-performing model on validation accuracy.

        **5. Inference pipeline (this app)**
        An uploaded image is resized to 224×224, converted to a NumPy array, and passed through
        ResNet50's `preprocess_input` before being scored by the model. The class with the highest
        softmax probability is shown as the prediction, alongside its confidence score and the
        top-5 most likely classes.
        """
    )

    st.subheader("Dataset")
    st.write(
        "Trained on the **PlantVillage** dataset, which contains labeled images of healthy "
        "and diseased leaves across multiple crop species, sourced from Kaggle."
    )
    st.link_button(
        "View dataset on Kaggle",
        "https://www.kaggle.com/datasets/mohitsingh1804/plantvillage",
    )

    st.subheader("Roadmap")
    st.markdown(
        """
        - 🌡️ Disease severity estimation
        - 💊 Treatment and pesticide recommendation system
        - 🔥 Grad-CAM visualization for explainability
        - 📱 Mobile deployment via TensorFlow Lite
        - 🎥 Real-time detection from a camera feed
        - 🌍 Multi-language support
        - 📄 Downloadable prediction reports
        """
    )

st.markdown(
    '<div class="footer-note">Plant Leaf Disease Detection · TensorFlow + Keras + Streamlit · '
    'Built for educational and research purposes</div>',
    unsafe_allow_html=True,
)