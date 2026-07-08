# 🏦 Customer Churn Prediction

A Machine Learning web application that predicts whether a bank customer is likely to leave the bank (Churn) or continue using its services based on customer information.

Built using **Python**, **Scikit-learn**, and **Streamlit**.

---

## 🚀 Live Demo
[Click Here](https://churn-prediction-model-ae5358rrq6sgzuoxxjgd9x.streamlit.app/)


---

## 📌 Project Overview

Customer churn prediction is an important business problem where companies try to identify customers who are likely to stop using their services.

This project uses Machine Learning to analyze customer details and predict whether a customer will:

- ✅ Stay with the bank
- ❌ Leave the bank (Churn)

---

## 📂 Dataset

The project uses the **Churn Modelling Dataset**, containing customer information such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

Target Variable:

- **Exited**
  - 0 → Customer stays
  - 1 → Customer leaves

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Pickle

---

## 📊 Machine Learning Workflow

1. Data Loading
2. Data Preprocessing
3. Removing unnecessary columns
4. Label Encoding
5. Train-Test Split
6. Feature Scaling
7. Model Training
8. Model Evaluation
9. Hyperparameter Tuning (GridSearchCV)
10. Model Saving
11. Streamlit Deployment

---

## 🤖 Models Used

- Decision Tree Classifier
- Random Forest Classifier ✅ (Best Performing Model)

The Random Forest model was selected after comparing model performance and further optimized using **GridSearchCV**.

---

## 📈 Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report

---

## 🌐 Streamlit Web App

The application allows users to enter customer details through an interactive interface and instantly predicts whether the customer is likely to churn.

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Customer-Churn-Prediction.git
```

Move into the project directory

```bash
cd Customer-Churn-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run Churn_app.py
```

---

## 🎯 Future Improvements

- Add probability/confidence score
- Improve UI with custom styling
- Deploy using Docker
- Feature importance visualization
- Support multiple ML models for comparison

---

## 👨‍💻 Author

**Prateek Verma**

GitHub: https://github.com/prateek2k5
---

⭐ If you found this project helpful, consider giving it a star!
