import streamlit as st
import joblib

# -----------------------------
# Page Setup
# -----------------------------

st.set_page_config(
    page_title="AI Review Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("sentiment-model/sentiment_model.pkl")
vectorizer = joblib.load("sentiment-model/tfidf_vectorizer.pkl")

# -----------------------------
# Simple Styling
# -----------------------------

st.markdown("""
<style>

.block-container {
    max-width: 800px;
    padding-top: 2rem;
    padding-bottom: 1rem;
}

h1 {
    text-align: center;
    font-size: 2.2rem !important;
    margin-bottom: 0.3rem !important;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.card {
    background-color: #f8f9ff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 12px;
    text-align: center;
}

.card-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: #4f46e5;
}

.card-label {
    font-size: 0.75rem;
    color: #6b7280;
}

.result {
    text-align: center;
    padding: 12px;
    border-radius: 12px;
    margin-top: 10px;
    font-size: 1.2rem;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# App Header
# -----------------------------

st.title("💬 AI Review Sentiment Analyzer")

st.markdown(
    '<div class="subtitle">'
    'Analyze customer reviews and predict whether the sentiment is positive or negative using Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Review Input
# -----------------------------

review = st.text_area(
    "✍️ Enter your customer review",
    placeholder="Example: I absolutely loved this product. The quality was amazing!",
    height=100
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("✨ Analyze Sentiment", use_container_width=True):

    if review.strip():

        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]

        if prediction == 1:

            st.markdown(
                '<div class="result" style="background:#ecfdf5; color:#047857;">'
                '😊 Positive Sentiment'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="result" style="background:#fff1f2; color:#be123c;">'
                '😡 Negative Sentiment'
                '</div>',
                unsafe_allow_html=True
            )

    else:

        st.warning("Please enter a review first.")


# -----------------------------
# Model Performance
# -----------------------------

st.markdown("### 📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-value">82%</div>
        <div class="card-label">Test Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-value">TF-IDF</div>
        <div class="card-label">Feature Extraction</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-value">Logistic Regression</div>
        <div class="card-label">ML Algorithm</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# Footer
# -----------------------------

st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.75rem; margin-top:20px;'>"
    "Built with Python • Scikit-learn • Streamlit"
    "</p>",
    unsafe_allow_html=True
)