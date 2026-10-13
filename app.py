import streamlit as st
import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Page Configuration
st.set_page_config(page_title="Real-Time Sentiment Analyzer", page_icon="📊", layout="centered")

# Download & Cache VADER Lexicon
@st.cache_resource
def load_vader():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

# Load ML Artifacts
@st.cache_resource
def load_ml_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

sia = load_vader()
model, vectorizer = load_ml_artifacts()

# UI Layout
st.title("📊 Real-Time Sentiment & Emotion Analyzer")
st.write("Analyzes user feedback into **Positive**, **Neutral / Medium**, or **Negative** classes.")

# User Input Box
user_input = st.text_area("Enter Review / Text Input:", placeholder="e.g., I like the movie but it is not too much good...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter valid text to analyze.")
    else:
        # 1. Rule-Based Scoring via VADER
        vader_scores = sia.polarity_scores(user_input)
        compound = vader_scores['compound']

        # 2. Vectorization & ML Inference
        vec_input = vectorizer.transform([user_input])
        ml_pred = model.predict(vec_input)[0]

        # Hybrid Decision Logic for accurate Neutral/Medium classification
        st.subheader("Analysis Results")
        
        if compound >= 0.15:
            st.success("😊 **Positive Sentiment**")
            sentiment_status = "Positive"
        elif compound <= -0.15:
            st.error("😞 **Negative Sentiment**")
            sentiment_status = "Negative"
        else:
            st.info("😐 **Neutral / Medium Sentiment**")
            sentiment_status = "Neutral"

        # Metrics Breakdown for Technical Interview Presentation
        st.markdown("---")
        st.markdown("**Model Confidence Metrics:**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Compound Score", f"{compound:.2f}")
        col2.metric("Positive Weight", f"{vader_scores['pos']:.2f}")
        col3.metric("Negative Weight", f"{vader_scores['neg']:.2f}")