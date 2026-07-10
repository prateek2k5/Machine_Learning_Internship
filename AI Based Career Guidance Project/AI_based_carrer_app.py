import streamlit as st
import pickle
import pandas as pd
import numpy as np

#Load Model
with open('ai_based_carrer_model.pkl','rb') as f:
    model=pickle.load(f)

#Heading of the Webpage
st.title("AI Based Carrer Prediction")
#Subheading of the Webpage
st.write("Select your skill level for each subject :")

# Dictionary used for Label Encoding
mapping = {
    "Average": 0,
    "Beginner": 1,
    "Excellent": 2,
    "Intermediate": 3,
    "Not Interested": 4,
    "Poor": 5
}

# User Inputs
database = st.selectbox(
    "Database Fundamentals",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

architecture = st.selectbox(
    "Computer Architecture",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

distributed = st.selectbox(
    "Distributed Computing Systems",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

cyber = st.selectbox(
    "Cyber Security",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

networking = st.selectbox(
    "Networking",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

software_dev = st.selectbox(
    "Software Development",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

programming = st.selectbox(
    "Programming Skills",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

project = st.selectbox(
    "Project Management",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

forensics = st.selectbox(
    "Computer Forensics Fundamentals",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

technical = st.selectbox(
    "Technical Communication",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

aiml = st.selectbox(
    "AI ML",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

software_eng = st.selectbox(
    "Software Engineering",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

business = st.selectbox(
    "Business Analysis",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

communication = st.selectbox(
    "Communication Skills",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

datascience = st.selectbox(
    "Data Science",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

troubleshooting = st.selectbox(
    "Troubleshooting Skills",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

graphics = st.selectbox(
    "Graphics Designing",
    ("Average","Beginner","Excellent","Intermediate","Not Interested","Poor"),
)

# Prediction Button
if st.button("Predict Career"):

    input_data = pd.DataFrame([{
        "Database Fundamentals": mapping[database],
        "Computer Architecture": mapping[architecture],
        "Distributed Computing Systems": mapping[distributed],
        "Cyber Security": mapping[cyber],
        "Networking": mapping[networking],
        "Software Development": mapping[software_dev],
        "Programming Skills": mapping[programming],
        "Project Management": mapping[project],
        "Computer Forensics Fundamentals": mapping[forensics],
        "Technical Communication": mapping[technical],
        "AI ML": mapping[aiml],
        "Software Engineering": mapping[software_eng],
        "Business Analysis": mapping[business],
        "Communication skills": mapping[communication],
        "Data Science": mapping[datascience],
        "Troubleshooting skills": mapping[troubleshooting],
        "Graphics Designing": mapping[graphics]
    }])

    prediction = model.predict(input_data)

    st.success(f"🎯 Recommended Career: {prediction[0]}")
