import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

# ==========================================
# Load Trained CNN Model
# ==========================================

# Saved CNN model load kar rahe hain.
model = load_model("cifar10_cnn.keras")

# ==========================================
# CIFAR-10 Class Names
# ==========================================

# Model output (0-9) ko readable names me convert karne ke liye.
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

# ==========================================
# App Title
# ==========================================

st.title("🖼️ CIFAR-10 Image Classification")
st.write(
    "Upload an image or capture one using your webcam to classify it using a trained CNN model."
)

# ==========================================
# Image Source Selection
# ==========================================

option = st.radio(
    "Select Image Source",
    (
        "📂 Upload Image",
        "📷 Capture from Camera"
    )
)

image = None

# ==========================================
# Upload Image
# ==========================================

if option == "📂 Upload Image":

    uploaded_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# ==========================================
# Capture Image
# ==========================================

else:

    camera_image = st.camera_input(
        "Take a Picture"
    )

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# ==========================================
# Prediction
# ==========================================

if image is not None:

    st.image(
        image,
        caption="Selected Image",
        use_container_width=True
    )

    # --------------------------------------
    # Image Preprocessing
    # --------------------------------------

    # Model ki input size 32x32 hai.
    image = image.resize((32, 32))

    # Image ko NumPy array me convert kar rahe hain.
    img_array = np.array(image)

    # Pixel values normalize kar rahe hain.
    img_array = img_array / 255.0

    # Batch dimension add kar rahe hain.
    img_array = np.expand_dims(img_array, axis=0)

    # ======================================
    # Predict Button
    # ======================================

    if st.button("🔍 Predict Image"):

        with st.spinner("Predicting..."):

            prediction = model.predict(
                img_array,
                verbose=0
            )

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction)

        # ==================================
        # Prediction Result
        # ==================================

        st.success(
            f"### ✅ Prediction : {class_names[predicted_class]}"
        )

        st.info(
            f"### 🎯 Confidence : {confidence*100:.2f}%"
        )

        # ==================================
        # Probability Chart
        # ==================================

        st.subheader("📊 Class Probabilities")

        for i, prob in enumerate(prediction[0]):

            st.write(f"**{class_names[i]}**")

            st.progress(float(prob))

            st.write(f"{prob*100:.2f}%")

# ==========================================
# Footer
# ==========================================

st.markdown("---")
st.caption("Developed using TensorFlow, Keras & Streamlit")
