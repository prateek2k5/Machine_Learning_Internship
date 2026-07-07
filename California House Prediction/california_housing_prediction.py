# ============================================================
# California Housing Price Prediction using Machine Learning
# ------------------------------------------------------------
# Author  : Prateek Verma
# Internship : Machine Learning with Python
# Week : 1
#
# Description:
# This project predicts California housing prices using
# different Regression Algorithms and compares their performance.
#
# Models Used:
# 1. Linear Regression
# 2. Polynomial Regression
# 3. Ridge Regression
# 4. Lasso Regression
# ============================================================


# ============================================================
# STEP 1 : Import Required Libraries
# ============================================================

# Numerical computations
import numpy as np

# Data manipulation
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Built-in California Housing Dataset
from sklearn.datasets import fetch_california_housing

# Splitting Dataset into Training and Testing Data
from sklearn.model_selection import train_test_split

# Feature Scaling and Polynomial Features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Evaluation Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ============================================================
# STEP 2 : Load California Housing Dataset
# ============================================================

print("=" * 70)
print("Loading California Housing Dataset...")
print("=" * 70)

# Load the dataset from Scikit-Learn
data = fetch_california_housing()

# Display dataset description
print(data.DESCR)

# Convert dataset into a Pandas DataFrame
df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# Add Target Column
df["MedHouseVal"] = data.target


# ============================================================
# STEP 3 : Explore the Dataset
# ============================================================

print("\nFirst Five Records")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())


# ============================================================
# STEP 4 : Exploratory Data Analysis (EDA)
# ============================================================

print("\nGenerating Correlation Heatmap...")

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.show()


# ============================================================
# STEP 5 : Separate Features and Target Variable
# ============================================================

# Features (Independent Variables)
X = df.drop("MedHouseVal", axis=1)

# Target (Dependent Variable)
y = df["MedHouseVal"]

print("\nFeature Matrix Shape :", X.shape)
print("Target Shape :", y.shape)


# ============================================================
# STEP 6 : Split Dataset
# ============================================================

print("\nSplitting Dataset into Training and Testing Data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])


# ============================================================
# STEP 7 : Feature Scaling
# ============================================================

print("\nScaling Features using StandardScaler...")

# Create StandardScaler Object
scaler = StandardScaler()

# Learn scaling parameters from training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply same transformation to test data
X_test_scaled = scaler.transform(X_test)

print("Feature Scaling Completed Successfully.")

print("=" * 70)
print("Data Preprocessing Completed")
print("=" * 70)

# ============================================================
# STEP 8 : Linear Regression Model
# ============================================================

print("\n" + "=" * 70)
print("Training Linear Regression Model...")
print("=" * 70)

# Create Linear Regression Model
lr = LinearRegression()

# Train the Model
lr.fit(X_train_scaled, y_train)

# Make Predictions
y_pred_lr = lr.predict(X_test_scaled)

# Calculate Evaluation Metrics
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Results")
print(f"RMSE     : {rmse_lr:.4f}")
print(f"R² Score : {r2_lr:.4f}")


# ============================================================
# STEP 9 : Polynomial Regression Model
# ============================================================

print("\n" + "=" * 70)
print("Training Polynomial Regression Model...")
print("=" * 70)

# Create Polynomial Features (Degree = 2)
poly = PolynomialFeatures(degree=2)

# Transform Training and Testing Data
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train Linear Regression on Polynomial Features
pr = LinearRegression()

pr.fit(X_train_poly, y_train)

# Predict
y_pred_pr = pr.predict(X_test_poly)

# Evaluation
rmse_pr = np.sqrt(mean_squared_error(y_test, y_pred_pr))
r2_pr = r2_score(y_test, y_pred_pr)

print("\nPolynomial Regression Results")
print(f"RMSE     : {rmse_pr:.4f}")
print(f"R² Score : {r2_pr:.4f}")


# ============================================================
# STEP 10 : Ridge Regression Model
# ============================================================

print("\n" + "=" * 70)
print("Training Ridge Regression Model...")
print("=" * 70)

# Create Ridge Model
ridge = Ridge(alpha=1.0)

# Train Model
ridge.fit(X_train_scaled, y_train)

# Predict
y_pred_ridge = ridge.predict(X_test_scaled)

# Evaluation
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nRidge Regression Results")
print(f"RMSE     : {rmse_ridge:.4f}")
print(f"R² Score : {r2_ridge:.4f}")


# ============================================================
# STEP 11 : Lasso Regression Model
# ============================================================

print("\n" + "=" * 70)
print("Training Lasso Regression Model...")
print("=" * 70)

# Create Lasso Model
lasso = Lasso(alpha=0.1)

# Train Model
lasso.fit(X_train_scaled, y_train)

# Predict
y_pred_lasso = lasso.predict(X_test_scaled)

# Evaluation
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\nLasso Regression Results")
print(f"RMSE     : {rmse_lasso:.4f}")
print(f"R² Score : {r2_lasso:.4f}")

# ============================================================
# STEP 12 : Compare Model Performance
# ============================================================

print("\n" + "=" * 70)
print("Comparing All Models")
print("=" * 70)

# Store model performance in a DataFrame
result = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Ridge Regression",
        "Lasso Regression"
    ],
    "RMSE": [
        rmse_lr,
        rmse_pr,
        rmse_ridge,
        rmse_lasso
    ],
    "R2 Score": [
        r2_lr,
        r2_pr,
        r2_ridge,
        r2_lasso
    ]
})

print(result)


# ============================================================
# STEP 13 : Save Results
# ============================================================

import os

# Create output folders if they don't exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("images", exist_ok=True)

# Save model comparison as CSV
result.to_csv("outputs/model_comparison.csv", index=False)

print("\nModel comparison saved successfully!")


# ============================================================
# STEP 14 : Save Evaluation Metrics
# ============================================================

with open("outputs/evaluation_metrics.txt", "w") as file:

    file.write("California Housing Price Prediction\n\n")

    file.write("Linear Regression\n")
    file.write(f"RMSE : {rmse_lr:.4f}\n")
    file.write(f"R2 Score : {r2_lr:.4f}\n\n")

    file.write("Polynomial Regression\n")
    file.write(f"RMSE : {rmse_pr:.4f}\n")
    file.write(f"R2 Score : {r2_pr:.4f}\n\n")

    file.write("Ridge Regression\n")
    file.write(f"RMSE : {rmse_ridge:.4f}\n")
    file.write(f"R2 Score : {r2_ridge:.4f}\n\n")

    file.write("Lasso Regression\n")
    file.write(f"RMSE : {rmse_lasso:.4f}\n")
    file.write(f"R2 Score : {r2_lasso:.4f}\n")

print("Evaluation metrics saved successfully!")


# ============================================================
# STEP 15 : Visualize Model Comparison
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(
    x="Model",
    y="R2 Score",
    data=result
)

plt.title("Model Comparison using R² Score")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("images/model_comparison.png")

plt.show()

print("Model comparison graph saved!")


# ============================================================
# STEP 16 : Actual vs Predicted (Best Model)
# ============================================================

# Assuming Polynomial Regression performed best

plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    y_pred_pr,
    alpha=0.5
)

plt.xlabel("Actual House Price")

plt.ylabel("Predicted House Price")

plt.title("Actual vs Predicted Values")

plt.tight_layout()

plt.savefig("images/actual_vs_predicted.png")

plt.show()

print("Actual vs Predicted graph saved!")


# ============================================================
# STEP 17 : Sample Prediction
# ============================================================

print("\n" + "=" * 70)
print("Prediction on New Data")
print("=" * 70)

# Take one sample from the testing dataset
testdf = X_test.tail(1)

# Scale the sample
testdf_scaled = scaler.transform(testdf)

# Polynomial transformation
testdf_poly = poly.transform(testdf_scaled)

# Predictions
prediction_linear = lr.predict(testdf_scaled)

prediction_polynomial = pr.predict(testdf_poly)

prediction_ridge = ridge.predict(testdf_scaled)

prediction_lasso = lasso.predict(testdf_scaled)

print("\nSample Input")
print(testdf)

print("\nPredicted House Price")

print(f"Linear Regression     : {prediction_linear[0]:.4f}")

print(f"Polynomial Regression : {prediction_polynomial[0]:.4f}")

print(f"Ridge Regression      : {prediction_ridge[0]:.4f}")

print(f"Lasso Regression      : {prediction_lasso[0]:.4f}")


# Save Prediction

with open("outputs/sample_prediction.txt","w") as file:

    file.write("Sample Prediction\n\n")

    file.write(f"Linear Regression : {prediction_linear[0]:.4f}\n")

    file.write(f"Polynomial Regression : {prediction_polynomial[0]:.4f}\n")

    file.write(f"Ridge Regression : {prediction_ridge[0]:.4f}\n")

    file.write(f"Lasso Regression : {prediction_lasso[0]:.4f}\n")

print("\nPrediction saved successfully!")


# ============================================================
# STEP 18 : Best Model
# ============================================================

best_model = result.loc[result["R2 Score"].idxmax()]

print("\n" + "=" * 70)
print("Best Performing Model")
print("=" * 70)

print(best_model)

print("\nProject Completed Successfully!")


#Deployment code
import joblib

joblib.dump(lr, "linear_model.pkl")
joblib.dump(scaler, "scaler.pkl")