import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

from src.preprocessing import clean_text
from src.features import sentiment_score, politeness_score, normalize_score
from src.semantic import semantic_bias
from src.mitigation import counterfactual
from src.bias_type import detect_bias_type

# -------------------------
# LOAD MODELS
# -------------------------
model = joblib.load("best_model.pkl")   # 🔥 use best model
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Bias Detection", layout="centered")

st.title("🧠 Bias Detection System")
st.write("Detect bias in workplace communication and generate neutral alternatives.")

text = st.text_area("✏️ Enter sentence:")

# -------------------------
# ANALYZE BUTTON
# -------------------------
if st.button("🔍 Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean = clean_text(text)

        # -------------------------
        # FEATURES
        # -------------------------
        X_tfidf = vectorizer.transform([clean])

        sem = semantic_bias(clean)
        sem_norm = normalize_score(sem)

        # extra feature must match training → only semantic used
        extra = np.array([[sem]])
        extra = scaler.transform(extra)

        X = hstack([X_tfidf, extra])

        # -------------------------
        # PREDICTION (HYBRID)
        # -------------------------
        prob = model.predict_proba(X)[0][1]

        final_score = 0.7 * prob + 0.3 * sem_norm

        pred = 1 if final_score > 0.4 else 0

        # -------------------------
        # ADDITIONAL ANALYSIS
        # -------------------------
        sentiment = sentiment_score(clean)
        politeness = politeness_score(clean)

        bias_type = detect_bias_type(clean, sentiment, politeness, sem)

        # -------------------------
        # OUTPUT
        # -------------------------
        st.subheader("📊 Results")

        if pred == 1:
            st.error("⚠️ Bias Detected")
        else:
            st.success("✅ Neutral Text")

        st.write(f"**Confidence Score:** {round(final_score, 2)}")
        st.write(f"**Bias Type:** {bias_type}")

        # -------------------------
        # MITIGATION (AUTO SHOW)
        # -------------------------
        if pred == 1:
            st.subheader("💡 Suggested Neutral Version")
            neutral = counterfactual(text)
            st.success(neutral)