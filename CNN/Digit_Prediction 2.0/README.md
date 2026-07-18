# ✍️ Handwritten Digit Recognition using CNN

A deep learning web application that recognizes handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.

The application is built with **TensorFlow**, **OpenCV**, and **Streamlit**, allowing users to predict handwritten digits through multiple input methods.

---

## 🚀 Live Demo

🔗 **Streamlit App**

https://digitpredictionapp20-9dyrgbahavzcjpvbnffkpx.streamlit.app/

---

## 🚀 Features

- 📁 Upload handwritten digit images
- ✏️ Draw digits directly on the canvas
- 📷 Capture digits using your camera
- 🧠 CNN-based handwritten digit recognition
- 🖼 Advanced OpenCV image preprocessing
- 📊 Prediction probability graph
- 🏆 Top-3 Predictions
- 📈 Confidence score
- 🎨 Interactive Streamlit interface

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy
- Matplotlib
- Pillow

---

## 📂 Project Structure

```text
Handwritten-Digit-Recognition-Using-CNN/
│
├── CNN_app.py
├── cnn_model.keras
├── requirements.txt
├── runtime.txt
├── README.md
└── sample_images/
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Handwritten-Digit-Recognition-Using-CNN.git
```

### 2. Move to the Project Directory

```bash
cd Handwritten-Digit-Recognition-Using-CNN
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run CNN_app.py
```

---

# ⚠️ Important

This project is built and tested using **Python 3.11**.

If you are using **VS Code**, make sure to manually select the **Python 3.11 interpreter** before running the project.

Go to:

```
Ctrl + Shift + P
→ Python: Select Interpreter
→ Select Python 3.11
```

Using Python 3.13 or newer versions may cause TensorFlow dependency issues and the project may not run correctly.

---

## 📷 Input Methods

- 📁 Upload Image
- ✏️ Draw Digit
- 📷 Camera Input

---

## 🖼 Image Preprocessing Pipeline

The uploaded image goes through multiple preprocessing steps before prediction:

- Grayscale Conversion
- Histogram Equalization
- Gaussian Blur
- Adaptive Thresholding
- Noise Removal
- Contour Detection
- Digit Cropping
- Aspect Ratio Preservation
- Center Alignment
- Normalization

Finally, the image is converted into a **28 × 28 MNIST-style image** and passed to the CNN model.

---

## 🧠 CNN Architecture

- Conv2D (32 Filters)
- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- Conv2D (64 Filters)
- MaxPooling2D
- Flatten
- Dense (256)
- Dropout
- Dense (10 Softmax)

---

## 📊 Application Output

The application displays:

- Predicted Digit
- Confidence Score
- Top-3 Predictions
- Prediction Probability Graph
- Probability Table

---

## 🔮 Future Improvements

- Multi-digit recognition
- Better preprocessing for real-world handwriting
- Mobile optimization
- Batch prediction
- Dark mode UI

---

## ⭐ If you like this project

Please consider giving this repository a **Star ⭐**.

---

# 👨‍💻 Developed By

## **Prateek**

**B.Tech Information Technology | AI & Machine Learning Enthusiast | Software Developer**

Thank you for visiting this repository! 🚀
