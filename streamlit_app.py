# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load word index and model
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
model = load_model('sentiment_analysis_model.h5')

# Step 2: Helper Functions
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Page Configuration and Styling
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
        }
        .stTextArea textarea {
            background-color: #1c1e26;
            color: #fff;
            font-size: 16px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 0.6em 1.5em;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .prediction {
            font-size: 24px;
            font-weight: 600;
            color: #F9C74F;
        }
        .score {
            font-size: 16px;
            color: #90BE6D;
        }
    </style>
""", unsafe_allow_html=True)

# Step 4: App Layout
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Enter a movie review and this model will predict whether it is **Positive** or **Negative**.")

# Input area
user_input = st.text_area("📽️ Movie Review", height=150, placeholder="Type your review here...")

# Button
if st.button("🚀 Classify Review"):
    if user_input.strip() == "":
        st.warning("Please enter a review before classifying.")
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😞"

        # Display results
        st.markdown(f"<div class='prediction'>Sentiment: {sentiment}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='score'>Prediction Confidence Score: {prediction[0][0]:.4f}</div>", unsafe_allow_html=True)
else:
    st.info("Enter a review and click **Classify Review** to get started!")

