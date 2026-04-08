import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix, hstack

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adverse Drug Reaction Predictor",
    page_icon="💊",
    layout="centered"
)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf = joblib.load("random_forest.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return rf, tfidf, feature_columns

try:
    rf, tfidf, feature_columns = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model files: {e}")

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("💊 Adverse Drug Reaction Predictor")
st.markdown("Predict whether a reported adverse drug reaction case is **serious or non-serious** based on patient information.")

st.divider()

st.subheader("Patient Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Patient Age (Years)", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female", "Unknown"])
    reporter_type = st.selectbox("Reporter Type", [
        "Healthcare Professional", "Consumer", "Physician",
        "Pharmacist", "Lawyer", "Not Specified"
    ])

with col2:
    weight = st.number_input("Patient Weight (kg)", min_value=0.0, max_value=300.0, value=70.0)
    report_source = st.selectbox("Report Source", [
        "Not Specified",
        "Health Professional",
        "Consumer",
        "Health Professional ,Literature",
        "Health Professional ,Foreign",
        "Consumer,Literature",
        "User Facility",
        "Other,Health Professional"
    ])

reactions = st.text_area(
    "Reported Reactions",
    placeholder="e.g. headache, fever, nausea",
    height=100
)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("Predict", type="primary", use_container_width=True):
    if not model_loaded:
        st.error("Model files not loaded. Make sure .pkl files are in the same directory.")
    else:
        # Build structured input
        input_dict = {
            "Patient Age": age,
            "Patient Weight": weight,
            "Sex": sex,
            "Reporter Type": reporter_type,
            "Report Source": report_source
        }

        input_df = pd.DataFrame([input_dict])
        input_df = pd.get_dummies(input_df)

        # Align columns to training features
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_columns].astype(float)

        # Build text input
        X_text = tfidf.transform([reactions])
        X_structured_sparse = csr_matrix(input_df.values)
        X_input = hstack([X_structured_sparse, X_text])

        # Predict
        pred = rf.predict(X_input)[0]
        prob = rf.predict_proba(X_input)[0]

        serious_prob = prob[1]
        non_serious_prob = prob[0]

        st.subheader("Prediction Result")

        if pred == 1:
            st.error(f"⚠️ **Serious Case** — {serious_prob:.1%} confidence")
        else:
            st.success(f"✅ **Non-Serious Case** — {non_serious_prob:.1%} confidence")

        # Probability bar
        st.markdown("**Probability Breakdown**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Serious", f"{serious_prob:.1%}")
        with col2:
            st.metric("Non-Serious", f"{non_serious_prob:.1%}")

        st.progress(float(serious_prob), text="Serious probability")

        st.caption("Model: Random Forest | Optimized for recall on serious cases (PR-AUC)")