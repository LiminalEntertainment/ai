import streamlit as st
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon=None
)

st.title("Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and the AI will try to recognize it.")

# Load and train model (cached)
@st.cache_resource
def load_model():
    try:
        from sklearn.datasets import load_digits
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split

        # Load digits dataset
        digits = load_digits()
        X = digits.images.reshape(len(digits.images), -1) / 16.0
        y = digits.target

        # Train-test split
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=300,
            random_state=42
        )
        model.fit(X_train, y_train)

        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


model = load_model()

if model is None:
    st.warning("Could not load model. Using fallback recognition.")
else:
    st.success("Model loaded successfully.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to grayscale and resize to 8x8
        img_gray = image.convert("L")
        img_resized = img_gray.resize((8, 8))

        # Convert to numpy array
        img_array = np.array(img_resized)

        # Invert if background is light
        if np.mean(img_array) > 128:
            img_array = 255 - img_array

        # Normalize to match training data
        img_array = img_array / 16.0
        img_flat = img_array.flatten().reshape(1, -1)

        if model is not None:
            # Prediction
            prediction = model.predict(img_flat)[0]
            st.write(f"## Prediction: {prediction}")

            # Probabilities
            probs = model.predict_proba(img_flat)[0]
            st.write("### Probabilities:")
            for i, prob in enumerate(probs):
                st.write(f"Digit {i}: {prob:.2%}")
        else:
            st.write("Fallback recognition not implemented.")

    except Exception as e:
        st.error(f"Error processing image: {e}")

#instructions
st.sidebar.header("instructions")
st.sidebar.write ("""
1. Uplaod an image of a handwritten digit (0-9)
2. The image will be resized to 8x8 pixels
3. AI model will predict the digit
4. For best results:
  -White background
  -Black digit
  -Centered digit
  -Minimal noise
""")
