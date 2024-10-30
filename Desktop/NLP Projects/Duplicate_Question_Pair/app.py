from helper import *
import streamlit as st


st.title("Question Similarity Predictor")

# Input fields for questions
question1 = st.text_input("Enter Question 1")
question2 = st.text_input("Enter Question 2")

if st.button("Predict"):
    result, probability = question_similarity_pipeline(question1, question2)
    st.write(f"Prediction: {result}")

