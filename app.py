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

# ── Drug list ─────────────────────────────────────────────────────────────────
DRUG_FLAG_COLS = [
    'has_Abraxane', 'has_Abstral', 'has_Accutane', 'has_Acetaminophen_And_Codeine',
    'has_Aciphex', 'has_Aclasta', 'has_Actemra', 'has_Actinomycin_D', 'has_Acupan',
    'has_Adcetris', 'has_Adderall', 'has_Advair_Hfa', 'has_Advil', 'has_Afinitor',
    'has_Aldactone', 'has_Alecensa', 'has_Alendronate', 'has_Alimta', 'has_Allegra',
    'has_Alunbrig', 'has_Amantadine.', 'has_Amitiza', 'has_Amlodipine',
    'has_Amoxicillin_Trihydrate', 'has_Ampicillin_And_Sulbactam', 'has_Androgel',
    'has_Atacand', 'has_Atrovent', 'has_Augmentin', 'has_Augtyro', 'has_Avastin',
    'has_Azulfidine_En_Tabs', 'has_Bactrim', 'has_Belsomra', 'has_Benadryl',
    'has_Benazepril', 'has_Benicar', 'has_Bio_Three', 'has_Cabometyx', 'has_Calonal',
    'has_Carafate', 'has_Cartia_Xt', 'has_Casodex', 'has_Cathflo_Activase',
    'has_Cefepime', 'has_Celebrex', 'has_Celexa', 'has_Clariscan', 'has_Colace',
    'has_Combivent', 'has_Comirnaty_Tozinameran', 'has_Compazine', 'has_Controloc',
    'has_Cordarone', 'has_Cotellic', 'has_Coumadin', 'has_Covid_19_Vaccine_Nos',
    'has_Creon', 'has_Crestor', 'has_Cymbalta', 'has_Cyramza', 'has_Cytoxan',
    'has_Dayvigo', 'has_Decadron', 'has_Dexilant', 'has_Diflucan', 'has_Dilantin',
    'has_Doliprane', 'has_Effexor', 'has_Eliquis', 'has_Eloxatin', 'has_Enbrel',
    'has_Enhertu', 'has_Epclusa', 'has_Erbitux', 'has_Erivedge', 'has_Esbriet',
    'has_Escitalopram', 'has_Esidrex', 'has_Farydak', 'has_Flovent', 'has_Fluoxetine',
    'has_Fragmin', 'has_Fycompa', 'has_Gaster', 'has_Gaviscon', 'has_Gavreto',
    'has_Gemzar', 'has_Harnal', 'has_Harvoni', 'has_Hemlibra', 'has_Herceptin',
    'has_Humalog', 'has_Humira', 'has_Hycamtin', 'has_Hydralazine', 'has_Ibrance',
    'has_Imbruvica', 'has_Imfinzi', 'has_Imodium', 'has_Inlyta', 'has_Innohep',
    'has_Iressa', 'has_Iscotin', 'has_Januvia', 'has_Jardiance', 'has_Keflex',
    'has_Keppra', 'has_Keytruda', 'has_Kisqali', 'has_Klonopin', 'has_Lamictal',
    'has_Lasix', 'has_Leucovorin', 'has_Leukine', 'has_Levaquin',
    'has_Levofloxacin_Hemihydrate', 'has_Levora', 'has_Lexapro', 'has_Linzess',
    'has_Lioresal', 'has_Lipiodol', 'has_Lipitor', 'has_Lomotil', 'has_Lorbrena',
    'has_Lovenox', 'has_Ludiomil', 'has_Lyrica', 'has_Macrobid', 'has_Marinol',
    'has_Medrol', 'has_Megace', 'has_Mekinist', 'has_Metformin', 'has_Methadone',
    'has_Methadose', 'has_Meticorten', 'has_Micardis', 'has_Miralax',
    'has_Mitoxantrone', 'has_Miya_Bm', 'has_Mobic', 'has_Moderna_Covid_19_Vaccine',
    'has_Mounjaro', 'has_Ms_Contin', 'has_Mvasi', 'has_Narcan', 'has_Nauzelin',
    'has_Navelbine', 'has_Neoral', 'has_Nexium', 'has_Ocrevus', 'has_Opdivo',
    'has_Oxycontin', 'has_Paraplatin', 'has_Paxil', 'has_Pepcid', 'has_Pepcid_Ac',
    'has_Percocet', 'has_Perjeta', 'has_Pfizer_Biontech_Covid_19_Vaccine',
    'has_Picillibacta', 'has_Piqray', 'has_Plaquenil', 'has_Plavix', 'has_Predonine',
    'has_Prevacid', 'has_Prilosec', 'has_Prilosec_Otc', 'has_Primperan',
    'has_Prinivil', 'has_Prohance', 'has_Prolia', 'has_Pulmicort_Turbuhaler',
    'has_Qinlock', 'has_Ranexa', 'has_Reflex', 'has_Refresh_Classic', 'has_Reglan',
    'has_Revlimid', 'has_Rheumatrex', 'has_Risperdal', 'has_Ritalin', 'has_Rituxan',
    'has_Rozlytrek', 'has_Senokot', 'has_Sinemet', 'has_Solu_Medrol', 'has_Suboxone',
    'has_Symproic', 'has_Tabrecta', 'has_Tafinlar', 'has_Tagrisso', 'has_Tamiflu',
    'has_Tarceva', 'has_Taxol', 'has_Taxotere', 'has_Tecentriq', 'has_Tegretol',
    'has_Topotecan', 'has_Torisel', 'has_Tramcet', 'has_Triumeq', 'has_Tums',
    'has_Tylenol', 'has_Tylenol_With_Codeine', 'has_Ultram', 'has_Unasyn',
    'has_Unituxin', 'has_Urso', 'has_Valium', 'has_Velcade', 'has_Vemlidy',
    'has_Venclexta', 'has_Venlafaxine', 'has_Venlafaxine_Hcl', 'has_Verapamil',
    'has_Vfend', 'has_Vicodin', 'has_Vimpat', 'has_Vinorelbine', 'has_Vitamin_C',
    'has_Vitamin_D', 'has_Vitamin_D3', 'has_Voltarene_Diclofenac_Sodium',
    'has_Votrient', 'has_Wellbutrin', 'has_Xalatan', 'has_Xalcom', 'has_Xalkori',
    'has_Xanax', 'has_Xarelto', 'has_Xeljanz', 'has_Xeloda', 'has_Xgeva',
    'has_Xtandi', 'has_Yervoy', 'has_Zarxio', 'has_Zelboraf', 'has_Zithromax',
    'has_Zocor', 'has_Zofran', 'has_Zoladex', 'has_Zoloft', 'has_Zometa',
    'has_Zonegran', 'has_Zykadia', 'has_Zyprexa', 'has_Zyrtec'
]

# Clean display names for the multiselect
DRUG_DISPLAY_NAMES = sorted([col.replace("has_", "").replace("_", " ") for col in DRUG_FLAG_COLS])

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("💊 Adverse Drug Reaction Predictor")
st.markdown("Predict whether a reported adverse drug reaction case is **serious or non-serious** based on patient information.")

st.divider()

st.subheader("Patient Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Patient Age (Years)", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female", "Unknown"])


with col2:
    weight = st.number_input("Patient Weight (kg)", min_value=0.0, max_value=300.0, value=70.0)
    
    reporter_type = st.selectbox("Reporter Type", [
        "Healthcare Professional",
        "Consumer",
        "Not Specified",
        "Other"
    ])
    
suspect_drugs = st.multiselect(
    "Suspect Drugs (select all that apply)",
    options=DRUG_DISPLAY_NAMES
)

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
        }

        # Add drug binary flags
        selected_flags = [f"has_{d.replace(' ', '_')}" for d in suspect_drugs]
        for flag in DRUG_FLAG_COLS:
            input_dict[flag] = 1 if flag in selected_flags else 0

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

        st.markdown("**Probability Breakdown**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Serious", f"{serious_prob:.1%}")
        with col2:
            st.metric("Non-Serious", f"{non_serious_prob:.1%}")

        st.progress(float(serious_prob), text="Serious probability")

        st.caption("Model: Random Forest | Optimized for recall on serious cases (PR-AUC)")