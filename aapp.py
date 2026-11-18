
import streamlit as st
import joblib
import re

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load model
try:
    model, tfidf = load_model()
except Exception:
    st.error("Model files missing! Check model.pkl and tfidf.pkl")
    st.stop()

# Website UI
st.title("📰 Fake News Detection System")

st.write(
    "Enter any news article below to check "
    "whether the model classifies it as Fake or Real."
)

st.info("Machine Learning Model: Logistic Regression")

# Input
news = st.text_area(
    "Enter News Article",
    height=200,
    placeholder="Paste your news article here..."
)

# Prediction button
if st.button("Check News", type="primary"):

    if news.strip() == "":
        st.warning("Please enter news first!")

    else:
        cleaned_news = clean_text(news)

        if not cleaned_news:
            st.warning("Please enter meaningful English text.")
        else:
            news_vector = tfidf.transform([cleaned_news])

            prediction = model.predict(news_vector)

            probabilities = model.predict_proba(news_vector)[0]

            if prediction[0] == 0:
                st.error("⚠️ FAKE NEWS")
            else:
                st.success("✅ REAL NEWS")

            fake_probability = probabilities[
                list(model.classes_).index(0)
            ] * 100

            real_probability = probabilities[
                list(model.classes_).index(1)
            ] * 100

            col1, col2 = st.columns(2)

            col1.metric(
                "Fake Probability",
                f"{fake_probability:.2f}%"
            )

            col2.metric(
                "Real Probability",
                f"{real_probability:.2f}%"
            )

            st.caption(
                "Probability scores are model estimates, "
                "not proof that a news article is true or false."
            )

st.divider()

st.write("Developed using Python, Streamlit and Machine Learning.")
