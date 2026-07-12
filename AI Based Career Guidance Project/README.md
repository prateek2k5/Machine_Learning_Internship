# 🎯 AI-Based Career Guidance System

An intelligent **Machine Learning-powered Career Guidance System** that recommends suitable career paths based on a user's technical skills and subject proficiency. The application uses a **Random Forest Classifier** trained on a career guidance dataset and is deployed using **Streamlit** for an interactive user experience.

## 🌐 Live Demo

🚀 **Try the Application:**
[AI-Based Career Guidance System](https://ai-based-career-guidance-mexpszbpqswyan2aue5aik.streamlit.app)

---

## 📌 Project Overview

Choosing the right career can be challenging for students and aspiring professionals. This project leverages Machine Learning to analyze a user's skill levels across multiple computer science domains and predicts the most suitable career role.

Users simply select their proficiency level in different subjects such as Programming Skills, AI & ML, Cyber Security, Data Science, Networking, Software Engineering, and more. The trained model then recommends an appropriate career path.

---

## ✨ Features

* 🎯 AI-powered career prediction
* 📊 Machine Learning-based recommendation system
* 🖥️ Interactive Streamlit web interface
* ⚡ Instant career prediction
* 📚 Skill-based analysis
* 🎨 User-friendly interface

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Streamlit**
* **Pickle**

---

## 📂 Project Structure

```text
AI-Based-Career-Guidance/
│
├── AI_based_carrer_app.py
├── ai_based_carrer_model.pkl
├── dataset9000.csv
├── requirements.txt
├── README.md
└── images/
```

---

## 📊 Machine Learning Workflow

1. Load the dataset
2. Data preprocessing
3. Label Encoding of categorical features
4. Train-Test Split
5. Train Random Forest Classifier
6. Evaluate model performance
7. Save trained model using Pickle
8. Deploy with Streamlit

---

## 📝 Input Features

The model uses the following features:

* Database Fundamentals
* Computer Architecture
* Distributed Computing Systems
* Cyber Security
* Networking
* Software Development
* Programming Skills
* Project Management
* Computer Forensics Fundamentals
* Technical Communication
* AI & ML
* Software Engineering
* Business Analysis
* Communication Skills
* Data Science
* Troubleshooting Skills
* Graphics Designing

Each feature is selected using one of the following proficiency levels:

* Average
* Beginner
* Excellent
* Intermediate
* Not Interested
* Poor

---

## 🚀 Installation

Clone the repository

```bash
git clone <your-github-repository-url>
```

Move into the project directory

```bash
cd AI-Based-Career-Guidance
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run AI_based_carrer_app.py
```

---

## 📈 Model Used

* **Random Forest Classifier**

---

## 📦 Libraries Used

* streamlit
* pandas
* numpy
* scikit-learn
* pickle

---

## 🎯 Future Improvements

* Confidence score for predictions
* Career roadmap generation
* Skill gap analysis
* Recommended learning resources
* Salary insights
* Resume analysis
* Career trend visualization

---

## 👨‍💻 Author

**Prateek Verma**

If you found this project useful, consider giving the repository a ⭐.
