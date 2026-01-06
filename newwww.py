import streamlit as st
import numpy as np
from PIL import Image

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Handwritten Digit Recognition (Learning AI)",
    page_icon="✍️"
)

st.title("Handwritten Digit Recognition")
st.write("Upload a handwritten digit. Correct the AI and it will learn over time.")

# ===============================
# Load & initialize model
# ===============================
@st.cache_resource
def load_model():
    from sklearn.datasets import load_digits
    from sklearn.neural_network import MLPClassifier

    digits = load_digits()
    X = digits.images.reshape(len(digits.images), -1) / 16.0
    y = digits.target

    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=1,
        random_state=42
    )

    # Initial training with all classes
    model.partial_fit(X, y, classes=np.arange(10))
    return model

model = load_model()
st.success("Model loaded and ready to learn.")

# ===============================
# File upload
# ===============================
uploaded_file = st.file_uploader(
    "Upload a handwritten digit image (0–9)",
    type=["png", "jpg", "jpeg"]
)

# ===============================
# Image processing & prediction
# ===============================
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to grayscale and resize
        img_gray = image.convert("L")
        img_resized = img_gray.resize((8, 8))

        img_array = np.array(img_resized)

        # Invert if background is light
        if np.mean(img_array) > 128:
            img_array = 255 - img_array

        # Normalize and flatten
        img_array = img_array / 16.0
        img_flat = img_array.flatten().reshape(1, -1)

        # Prediction
        prediction = model.predict(img_flat)[0]
        probs = model.predict_proba(img_flat)[0]

        st.markdown(f"## 🧠 Prediction: **{prediction}**")

        st.write("### Confidence:")
        for i, p in enumerate(probs):
            st.write(f"Digit {i}: {p:.2%}")

        # ===============================
        # Feedback / Learning
        # ===============================
        st.divider()
        st.subheader("Teach the AI")

        correct_label = st.number_input(
            "What digit is this really?",
            min_value=0,
            max_value=9,
            step=1
        )

        if st.button("Reward / Update Model"):
            model.partial_fit(
                img_flat,
                [correct_label],
                classes=np.arange(10)   # 🔑 REQUIRED FIX
            )
            st.success("✅ AI learned from this example.")

    except Exception as e:
        st.error(f"Error processing image: {e}")

# ===============================
# Sidebar instructions
# ===============================
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload a handwritten digit (0–9)
2. AI predicts the digit
3. If wrong, enter the correct one
4. Click **Reward / Update Model**
5. The AI improves over time

**Tips:**
- White background
- Black digit
- Centered
- Minimal noise
""")
