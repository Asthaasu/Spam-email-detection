import streamlit as st
import pickle
from src.data_prep import preprocess

# Load model
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
nb_model = pickle.load(open("model/naive_bayes.pkl", "rb"))

# Page config
st.set_page_config(page_title="Spam Detector", page_icon="📧")

# Title
st.title("📧 Spam Email Detector")
st.write("Paste an email and check if it's spam")

# Input
email_input = st.text_area("Enter email text")

# Predict button
if st.button("Predict"):
    if email_input.strip() == "":
        st.warning("Please enter email text")
    else:
        # Preprocess input
        cleaned = preprocess(email_input)
        transformed = vectorizer.transform([cleaned])

        st.write("### 🔍 Cleaned Text")
        st.code(cleaned)

        # Prediction
        pred = nb_model.predict(transformed)[0]
        prob = nb_model.predict_proba(transformed)[0]

        # Output
        st.write("## 🎯 Prediction")

        if pred == 1:
            st.error(f"🚨 Spam Email (Confidence: {max(prob):.2f})")
        else:
            st.success(f"✅ Not Spam (Confidence: {max(prob):.2f})")

        # Insight
        st.write("### 💡 How it works")
        st.info(
            "This model uses TF-IDF to convert text into numerical features and "
            "Naive Bayes to classify emails based on word probabilities."
        )
