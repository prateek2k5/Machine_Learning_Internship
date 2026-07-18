# ============================================================
#        HANDWRITTEN DIGIT RECOGNITION USING CNN
# ============================================================

# ============================================================
#                    IMPORT LIBRARIES
# ============================================================

# Streamlit library
# Iska use web application banane ke liye hota hai.
import streamlit as st

# NumPy
# Numerical operations aur arrays handle karne ke liye.
import numpy as np

# OpenCV
# Image preprocessing aur computer vision operations ke liye.
import cv2

# Pillow (PIL)
# Uploaded images ko open/read karne ke liye.
from PIL import Image

# TensorFlow
# Trained CNN model load karne aur prediction karne ke liye.
import tensorflow as tf

# Matplotlib
# Prediction probabilities ka graph banane ke liye.
import matplotlib.pyplot as plt

# Streamlit Drawable Canvas
# Browser ke andar digit draw karne ke liye.
from streamlit_drawable_canvas import st_canvas


# ============================================================
#                 STREAMLIT PAGE CONFIGURATION
# ============================================================

# Ye browser tab ka title, icon aur layout set karta hai.
st.set_page_config(

    page_title="Handwritten Digit Recognition",

    page_icon="✍️",

    layout="centered"

)


# ============================================================
#                    LOAD CNN MODEL
# ============================================================

# Cache resource ka use isliye kiya gaya hai
# taaki model baar-baar reload na ho.
# Isse application fast chalti hai.

@st.cache_resource
def load_model():

    # Trained CNN model load kar rahe hain.
    model = tf.keras.models.load_model("cnn_model.keras")

    return model


# Function call
model = load_model()


# ============================================================
#                     PAGE TITLE
# ============================================================

st.title("✍️ Handwritten Digit Recognition")

st.markdown(
"""
### Welcome 👋

Is application ki help se aap kisi bhi handwritten digit ko predict
kar sakte hain.

Supported digits:

**0 1 2 3 4 5 6 7 8 9**

Input dene ke 3 methods available hain:

- 📁 Upload Image
- ✏ Draw Digit
- 📷 Camera

CNN model image ko preprocess karega aur final prediction dega.
"""
)


# ============================================================
#                  SIDEBAR INPUT OPTIONS
# ============================================================

st.sidebar.header("Choose Input Method")

option = st.sidebar.radio(

    "Select One",

    (

        "Upload Image",

        "Draw Digit",

        "Camera"

    )

)

# Variable initialize kar rahe hain.
# Isme final image store hogi.
image = None


# ============================================================
#                  OPTION 1 : UPLOAD IMAGE
# ============================================================

if option == "Upload Image":

    uploaded_file = st.file_uploader(

        "Upload an Image",

        type=["png", "jpg", "jpeg"]

    )

    if uploaded_file is not None:

        # PIL image open kar rahe hain.
        image = Image.open(uploaded_file)

        # RGB me convert kar rahe hain.
        image = image.convert("RGB")

        # User ko uploaded image show kar rahe hain.
        st.image(

            image,

            caption="Uploaded Image",

            use_container_width=True

        )

        # NumPy array me convert kar rahe hain.
        image = np.array(image)


# ============================================================
#                  OPTION 2 : DRAW DIGIT
# ============================================================

elif option == "Draw Digit":

    st.write("### Draw your digit below 👇")

    canvas_result = st_canvas(

        fill_color="black",

        stroke_width=10,

        stroke_color="white",

        background_color="black",

        height=280,

        width=280,

        drawing_mode="freedraw",

        key="canvas"

    )

    # Agar user ne kuch draw kiya hai.
    if canvas_result.image_data is not None:

        image = canvas_result.image_data.astype(np.uint8)

        st.image(

            image,

            caption="Drawn Digit",

            use_container_width=True

        )


# ============================================================
#                  OPTION 3 : CAMERA INPUT
# ============================================================

elif option == "Camera":

    picture = st.camera_input(

        "Capture Digit"

    )

    if picture is not None:

        image = Image.open(picture)

        image = image.convert("RGB")

        st.image(

            image,

            caption="Captured Image",

            use_container_width=True

        )

        image = np.array(image)


# ============================================================
#          IMAGE AVAILABLE CHECK
# ============================================================

# Agar image available hai tabhi preprocessing start hogi.
# Agar image None hai to niche wala code execute nahi hoga.

if image is not None:
        # ============================================================
    #                IMAGE PREPROCESSING
    # ============================================================

    # ============================================================
    # STEP 1 : CONVERT IMAGE TO GRAYSCALE
    # ============================================================

    # CNN model MNIST dataset par train hua tha.
    # MNIST images sirf grayscale hoti hain.
    # Isliye RGB image ko grayscale me convert karna zaruri hai.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Original grayscale image show kar rahe hain.
    st.subheader("Step 1 : Grayscale Image")

    st.image(

        gray,

        clamp=True,

        use_container_width=True

    )



    # ============================================================
    # STEP 2 : HISTOGRAM EQUALIZATION
    # ============================================================

    # Histogram Equalization image ka contrast improve karta hai.
    # Agar image dark ya bright ho to digit aur clearly dikhegi.

    gray = cv2.equalizeHist(gray)

    st.subheader("Step 2 : Histogram Equalization")

    st.image(

        gray,

        clamp=True,

        use_container_width=True

    )



    # ============================================================
    # STEP 3 : GAUSSIAN BLUR
    # ============================================================

    # Gaussian Blur unwanted noise remove karta hai.
    # Ye handwriting ko smooth banata hai.

    gray = cv2.GaussianBlur(

        gray,

        (5, 5),

        0

    )

    st.subheader("Step 3 : Gaussian Blur")

    st.image(

        gray,

        clamp=True,

        use_container_width=True

    )



    # ============================================================
    # STEP 4 : ADAPTIVE THRESHOLD
    # ============================================================

    # Adaptive Threshold har region ke according threshold choose karta hai.
    # Isse shadows aur uneven lighting ka effect kam ho jata hai.

    thresh = cv2.adaptiveThreshold(

        gray,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY_INV,

        11,

        2

    )

    st.subheader("Step 4 : Adaptive Threshold")

    st.image(

        thresh,

        clamp=True,

        use_container_width=True

    )



    # ============================================================
    # STEP 5 : MORPHOLOGICAL OPENING
    # ============================================================

    # Morphological Opening
    # Small white dots aur unwanted noise remove karta hai.

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

    st.subheader("Step 5 : Noise Removal")

    st.image(

        thresh,

        clamp=True,

        use_container_width=True

    )



    # ============================================================
    # STEP 6 : FIND CONTOURS
    # ============================================================

    # Contours image ke white objects detect karte hain.
    # Hume handwritten digit ka contour chahiye.

    contours, _ = cv2.findContours(

        thresh,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )



    # ============================================================
    # REMOVE VERY SMALL CONTOURS
    # ============================================================

    # Bahut chhote contours mostly noise hote hain.

    contours = [

        contour

        for contour in contours

        if cv2.contourArea(contour) > 300

    ]



    # ============================================================
    # IF NO DIGIT FOUND
    # ============================================================

    if len(contours) == 0:

        st.error(

            "❌ No handwritten digit detected."

        )

        st.stop()



    # ============================================================
    # SELECT LARGEST CONTOUR
    # ============================================================

    # Largest contour ko handwritten digit maan rahe hain.

    contour = max(

        contours,

        key=cv2.contourArea

    )



    # ============================================================
    # BOUNDING RECTANGLE
    # ============================================================

    # Digit ke around rectangle banaya ja raha hai.

    x, y, w, h = cv2.boundingRect(

        contour

    )



    # ============================================================
    # CROP ONLY DIGIT
    # ============================================================
    
    #Padding add kar rahe hain taaki digit ka complete part mile.
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(thresh.shape[1], x + w + padding)
    y2 = min(thresh.shape[0], y + h + padding)

    digit = thresh[y1:y2, x1:x2]

    st.subheader("Step 6 : Cropped Digit")

    st.image(

        digit,

        clamp=True,

        width=200

    )



    # ============================================================
    # STEP 7 : RESIZE WHILE MAINTAINING ASPECT RATIO
    # ============================================================

    # Digit ko directly 28x28 resize nahi karte.
    # Aspect ratio maintain karna important hai.
    # Warna digit stretch ho sakti hai.

    height, width = digit.shape

    if height > width:

        new_height = 20

        new_width = max(

            1,

            int(width * 20 / height)

        )

    else:

        new_width = 20

        new_height = max(

            1,

            int(height * 20 / width)

        )



    digit = cv2.resize(

        digit,

        (new_width, new_height),

        interpolation=cv2.INTER_AREA

    )



    # ============================================================
    # STEP 8 : CREATE BLACK 28x28 CANVAS
    # ============================================================

    # MNIST images exactly 28x28 hoti hain.

    canvas = np.zeros(

        (28, 28),

        dtype=np.uint8

    )



    # ============================================================
    # CENTER DIGIT ON CANVAS
    # ============================================================

    x_offset = (28 - new_width) // 2

    y_offset = (28 - new_height) // 2

    canvas[

        y_offset:y_offset+new_height,

        x_offset:x_offset+new_width

    ] = digit



    # ============================================================
    # STEP 9 : RECENTER DIGIT
    # ============================================================

    # Digit ko aur accurately center karte hain.

    coordinates = cv2.findNonZero(

        canvas

    )

    if coordinates is not None:

        x, y, w, h = cv2.boundingRect(

            coordinates

        )

        digit = canvas[

            y:y+h,

            x:x+w

        ]

        canvas = np.zeros(

            (28, 28),

            dtype=np.uint8

        )

        start_x = (28 - w) // 2

        start_y = (28 - h) // 2

        canvas[

            start_y:start_y+h,

            start_x:start_x+w

        ] = digit



    # ============================================================
    # SHOW FINAL PROCESSED IMAGE
    # ============================================================

    st.subheader("Final MNIST Style Image")

    st.image(

        canvas,

        clamp=True,

        width=220

    )



    # ============================================================
    # STEP 10 : NORMALIZATION
    # ============================================================

    # Pixel values 0-255 se 0-1 range me convert kar rahe hain.

    img_array = canvas.astype(

        "float32"

    ) / 255.0



    # ============================================================
    # FINAL CNN INPUT SHAPE
    # ============================================================

    # CNN ko input shape chahiye:
    # (Batch Size, Height, Width, Channel)

    img_array = img_array.reshape(

        1,

        28,

        28,

        1

    )

        # ============================================================
    #                 CNN MODEL PREDICTION
    # ============================================================

    # Preprocessed image ko trained CNN model me bhej rahe hain.
    # verbose=0 ka matlab prediction silently hogi.
    prediction = model.predict(

        img_array,

        verbose=0

    )

    # Prediction shape (1,10) hoti hai.
    # Hume sirf first row chahiye.
    prediction = prediction[0]



    # ============================================================
    #            TOP 3 MOST CONFIDENT PREDICTIONS
    # ============================================================

    # np.argsort() probabilities ko ascending order me sort karta hai.
    # [::-1] us order ko reverse karke descending bana deta hai.
    # [:3] sirf top 3 indices return karta hai.

    top3 = np.argsort(

        prediction

    )[::-1][:3]



    # ============================================================
    #              FINAL PREDICTED DIGIT
    # ============================================================

    # Highest probability wala index hi predicted digit hai.

    predicted_digit = top3[0]



    # ============================================================
    #              CONFIDENCE SCORE
    # ============================================================

    # Highest probability ko percentage me convert kar rahe hain.

    confidence = prediction[predicted_digit] * 100



    # ============================================================
    #                 PREDICTION RESULT
    # ============================================================

    st.success(

        "Prediction Completed Successfully ✅"

    )



    # ============================================================
    #                 METRIC CARDS
    # ============================================================

    # Do columns bana rahe hain.

    col1, col2 = st.columns(2)



    # ------------------------------------------------------------
    # Predicted Digit
    # ------------------------------------------------------------

    with col1:

        st.metric(

            label="Predicted Digit",

            value=str(predicted_digit)

        )



    # ------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------

    with col2:

        st.metric(

            label="Confidence",

            value=f"{confidence:.2f}%"

        )



    # ============================================================
    #                  TOP 3 PREDICTIONS
    # ============================================================

    st.subheader("🏆 Top 3 Predictions")

    for rank, digit in enumerate(

        top3,

        start=1

    ):

        st.write(

            f"**{rank}. Digit {digit}**  →  **{prediction[digit]*100:.2f}%**"

        )



    # ============================================================
    #            BAR GRAPH OF ALL PROBABILITIES
    # ============================================================

    st.subheader("Prediction Probability Graph")



    fig, ax = plt.subplots(

        figsize=(8,4)

    )



    ax.bar(

        range(10),

        prediction

    )



    ax.set_xticks(

        range(10)

    )



    ax.set_xlabel(

        "Digits"

    )



    ax.set_ylabel(

        "Probability"

    )



    ax.set_title(

        "CNN Prediction Probabilities"

    )



    st.pyplot(fig)



    # ============================================================
    #                PROBABILITY TABLE
    # ============================================================

    st.subheader("Probability Table")



    probability_dict = {

        str(i): f"{prediction[i]*100:.2f}%"

        for i in range(10)

    }



    st.table(

        probability_dict

    )



    # ============================================================
    #              CONFIDENCE INTERPRETATION
    # ============================================================

    # Confidence ke basis par user ko message dikhaya ja raha hai.

    if confidence >= 99:

        st.success("""
Excellent Prediction ✅

Model is extremely confident.
            """

        )



    elif confidence >= 95:

        st.info(

            """
Very High Confidence 👍

Prediction is highly reliable.
            """

        )



    elif confidence >= 85:

        st.warning(

            """
Good Prediction ⚡

Image thodi noisy ho sakti hai.
            """

        )



    elif confidence >= 70:

        st.warning(

            """
Average Confidence

Better handwriting se aur accurate prediction milegi.
            """

        )



    else:

        st.error(

            """
Low Confidence ❌

Try:

• Dark handwriting

• White background

• Single digit only

• Centered digit

• Proper lighting
            """

        )

# ============================================================
#                     SIDEBAR INFORMATION
# ============================================================

# Sidebar ka title
st.sidebar.title("ℹ️ About This Project")

# Sidebar me project ki basic information show kar rahe hain.
st.sidebar.info(
"""
# ✍️ Handwritten Digit Recognition

This application uses a **Convolutional Neural Network (CNN)**
trained on the famous **MNIST Dataset** to recognize
handwritten digits from **0 to 9**.

---

## 🚀 Features

✅ Upload Image

✅ Draw Digit

✅ Camera Input

✅ Advanced Image Preprocessing

✅ MNIST Style Centering

✅ CNN Prediction

✅ Top-3 Predictions

✅ Probability Graph

✅ Confidence Score

---

## 🧠 Deep Learning Model

- TensorFlow
- Keras
- CNN
- OpenCV
- Streamlit

---
"""
)


# ============================================================
#                MODEL INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Model Information")

st.sidebar.write("**Dataset :** MNIST")

st.sidebar.write("**Input Size :** 28 × 28")

st.sidebar.write("**Classes :** 10 Digits")

st.sidebar.write("**Framework :** TensorFlow")

st.sidebar.write("**Architecture :** CNN")


# ============================================================
#               IMAGE REQUIREMENTS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📷 Best Results")

st.sidebar.write("✔ Single handwritten digit")

st.sidebar.write("✔ White background")

st.sidebar.write("✔ Dark handwriting")

st.sidebar.write("✔ Digit should be centered")

st.sidebar.write("✔ Avoid multiple digits")

st.sidebar.write("✔ Avoid blurry images")


# ============================================================
#                PERFORMANCE TIPS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("💡 Tips")

st.sidebar.write(
"""
• Write digits clearly.

• Avoid touching image boundaries.

• Keep only one digit.

• Upload good quality images.

• Use proper lighting while capturing image.
"""
)


# ============================================================
#                 APPLICATION DETAILS
# ============================================================

st.markdown("---")

st.subheader("📌 About This Application")

st.write(
"""
This application demonstrates how **Deep Learning** and
**Computer Vision** can work together for handwritten
digit recognition.

The uploaded image first goes through several preprocessing
steps like:

- Grayscale Conversion
- Histogram Equalization
- Gaussian Blur
- Adaptive Thresholding
- Noise Removal
- Digit Cropping
- Aspect Ratio Preservation
- Center Alignment
- Normalization

Finally, the processed image is passed to a trained
**Convolutional Neural Network (CNN)** which predicts
the handwritten digit.
"""
)


# ============================================================
#                 TECHNOLOGIES USED
# ============================================================

st.markdown("---")

st.subheader("🛠 Technologies Used")

st.write(
"""
- Python

- Streamlit

- TensorFlow

- Keras

- OpenCV

- NumPy

- Matplotlib

- Pillow
"""
)


# ============================================================
#                  FOOTER
# ============================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;'>

### ✍️ Handwritten Digit Recognition using CNN

Developed with ❤️ using

Python • TensorFlow • OpenCV • Streamlit

</div>
""",
unsafe_allow_html=True
)


# ============================================================
#                 DEVELOPER CREDIT
# ============================================================

st.markdown(
"""
<div style='text-align:center; font-size:18px;'>

👨‍💻 Developed by <b>Prateek</b>

</div>
""",
unsafe_allow_html=True
)