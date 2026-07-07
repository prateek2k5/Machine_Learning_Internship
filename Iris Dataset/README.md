# 🌸 Iris Flower Classification using Machine Learning

An interactive **Machine Learning web application** built with **Python** and **Streamlit** that predicts the species of an Iris flower based on four botanical measurements. The application uses a **Random Forest Classifier** trained on the famous **Iris Dataset** from **scikit-learn** and saves the trained model using **Pickle**.

---

## 🚀 Live Demo

🔗 **Try the Application Here:**  
https://iris-dataset-project-krhicelxcjchvkmndavhk3.streamlit.app/

---

## 📌 Project Overview

This project demonstrates the end-to-end workflow of a Machine Learning classification problem:

- Load the Iris dataset from Scikit-learn
- Split the dataset into training and testing sets
- Train a Random Forest Classifier
- Save the trained model using Pickle
- Build an interactive web interface using Streamlit
- Deploy the application on Streamlit Community Cloud

---

## ✨ Features

- 🌸 Predicts Iris flower species
- 🎛️ Interactive slider-based input
- 🤖 Random Forest Classification model
- ⚡ Real-time predictions
- 💾 Model serialization using Pickle
- ☁️ Deployed on Streamlit Community Cloud
- 📱 Simple and responsive interface

---

## 📊 Dataset Information

The project uses the built-in **Iris Dataset** from **scikit-learn**.

| Property | Value |
|----------|-------|
| Samples | 150 |
| Features | 4 |
| Classes | 3 |

### Input Features

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

### Output Classes

- 🌸 Setosa
- 🌺 Versicolor
- 🌼 Virginica

---

## 🧠 Machine Learning Model

**Algorithm Used**

- Random Forest Classifier

### Workflow

```text
Load Dataset
      │
      ▼
Train-Test Split
      │
      ▼
Train Random Forest Model
      │
      ▼
Save Model (.pkl)
      │
      ▼
Build Streamlit App
      │
      ▼
Predict Iris Species
```

---

## 🛠️ Tech Stack

- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Pickle

---

## 📂 Project Structure

```
Iris-Dataset-Project/
│
├── app.py
├── iris_model.pkl
├── Iris_Dataset.ipynb
├── App.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Iris-Dataset-Project.git
cd Iris-Dataset-Project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 🎮 How to Use

1. Open the deployed Streamlit application.
2. Adjust the four flower measurements using the sliders.
3. Click the **Predict** button.
4. View the predicted Iris flower species instantly.

---

## 📦 Requirements

```
streamlit
numpy
pandas
scikit-learn
```

---

## 🎯 Learning Outcomes

Through this project, I learned:

- Machine Learning Classification
- Random Forest Classifier
- Train-Test Split
- Working with Scikit-learn datasets
- Model Serialization using Pickle
- Building interactive applications with Streamlit
- Deploying Machine Learning models to the cloud

---

## 👨‍💻 Author

**Prateek Verma**

B.Tech Information Technology Student | Machine Learning Enthusiast | Python Developer

---

⭐ **If you like this project, don't forget to give this repository a Star!**
