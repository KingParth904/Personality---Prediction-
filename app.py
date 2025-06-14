import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/personality_model.pkl")

st.title("Personality Predictor")

Time_spent_Alone = st.slider("Time spent alone", 0, 10)
Stage_fear = st.selectbox("Stage fear", ["Yes", "No"])
Social_event_attendance = st.slider("Social event attendance", 0, 10)
Going_outside = st.slider("Going outside", 0, 10)
Drained = st.selectbox("Drained after socializing", ["Yes", "No"])
Friends_circle_size = st.slider("Friends circle size", 0, 20)
Post_frequency = st.slider("Post frequency", 0, 10)

Stage_fear = 1 if Stage_fear == "Yes" else 0
Drained = 1 if Drained == "Yes" else 0

features = np.array([[Time_spent_Alone, Stage_fear, Social_event_attendance,
                      Going_outside, Drained, Friends_circle_size, Post_frequency]])

if st.button("Predict"):
    pred = model.predict(features)[0]
    st.success(f"Predicted Personality: {'Extrovert' if pred else 'Introvert'}")
