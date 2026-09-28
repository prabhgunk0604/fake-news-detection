"""
FAKE NEWS DETECTION - Streamlit Web App (Styled like modern AI dashboard)
----------------------------------------------------------------------------
Run this with:  streamlit run app.py
"""

import re
import string
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fake News Detector", page_icon="🛡️", layout="wide")

# ---------- Custom CSS (matches the uploaded dashboard design) ----------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #eef2ff, #f8f9ff, #e0e7ff);
    }

    /* ---------- Navbar ---------- */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 30px;
        background: rgba(255,255,255,0.75);
        border-radius: 16px;
        border: 1px solid rgba(99,102,241,0.15);
        margin-bottom: 30px;
    }
    .logo { font-size: 22px; font-weight: 700; color: #4f46e5; }
    .logo span { color: #111827; }
    .nav-links { color: #475569; font-weight: 500; }

    /* ---------- Badge ---------- */
    .badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 50px;
        background: #e0e7ff;
        color: #4338ca;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    /* ---------- Hero text ---------- */
    .hero-title {
        font-size: 42px;
        font-weight: 700;
        color: #111827;
        line-height: 1.15;
        margin-bottom: 15px;
    }
    .hero-title span { color: #4f46e5; }
    .hero-sub {
        font-size: 16px;
        color: #64748b;
        line-height: 1.6;
        margin-bottom: 10px;
    }

    /* ---------- AI card (decorative, right side of hero) ---------- */
    .ai-card {
        padding: 30px;
        background: rgba(255,255,255,0.85);
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,0.8);
        box-shadow: 0 20px 45px rgba(15,23,42,0.10);
        text-align: center;
    }
    .ai-icon {
        width: 65px; height: 65px;
        margin: auto; margin-bottom: 15px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 50%;
        background: linear-gradient(135deg, #4f46e5, #9333ea);
        color: white; font-size: 28px;
    }
    .ai-card h3 { color: #111827; margin-bottom: 6px; }
    .ai-card p { color: #64748b; font-size: 14px; }

    /* ---------- Detector box ---------- */
    .detector-box {
        padding: 30px;
        background: white;
        border-radius: 22px;
        box-shadow: 0 15px 40px rgba(15,23,42,0.08);
        margin-top: 10px;
        margin-bottom: 25px;
    }
    .section-title h2 { font-size: 28px; color: #111827; text-align:center; margin-bottom: 5px;}
    .section-title p { color: #64748b; text-align:center; margin-bottom: 20px;}

    /* Streamlit widget overrides */
    .stTextArea textarea {
        border: 2px solid #e2e8f0 !important;
        border-radius: 15px !important;
        font-size: 15px !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99,102,241,0.10) !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        font-weight: 600;
        font-size: 16px;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        box-shadow: 0 10px 25px rgba(79,70,229,0.25);
        transition: 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px rgba(79,70,229,0.35);
    }

    /* ---------- Result card ---------- */
    .result-card {
        margin-top: 20px;
        padding: 22px;
        border-radius: 16px;
        background: #f8fafc;
        border-left: 5px solid #4f46e5;
    }
    .result-real { color: #16a34a; font-size: 24px; font-weight: 700; }
    .result-fake { color: #dc2626; font-size: 24px; font-weight: 700; }

    /* ---------- Feature cards ---------- */
    .feature-card {
        padding: 25px;
        background: white;
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(15,23,42,0.07);
        text-align: center;
        height: 100%;
    }
    .feature-icon { font-size: 28px; margin-bottom: 10px; }
    .feature-card h4 { color: #111827; margin-bottom: 6px; }
    .feature-card p { color: #64748b; font-size: 14px; line-height: 1.5; }

    /* ---------- Stats ---------- */
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(15,23,42,0.06);
    }
    .stat-card h2 { color: #4f46e5; font-size: 28px; margin: 0; }
    .stat-card p { color: #64748b; margin-top: 4px; font-size: 13px; }

    /* ---------- Footer ---------- */
    .footer-box {
        margin-top: 40px;
        padding: 22px;
        text-align: center;
        background: #111827;
        color: #cbd5e1;
        border-radius: 16px;
        font-size: 13px;
    }
    .footer-box span { color: #818cf8; font-weight: 600; }

    footer, .stDeployButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

def clean_text(text):
    text = str(text)
    # Kaggle dataset ka "shortcut" hatao: real news mein "WASHINGTON (Reuters) -" jaisa tag hota hai
    text = re.sub(r"^.{0,80}?\(reuters\)\s*-?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(reuters|getty images|featured image|21st century wire|21wire)\b", " ", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- Navbar ----------
st.markdown("""
<div class="navbar">
    <div class="logo">🛡️ Truth<span>Check</span></div>
    <div class="nav-links">AI That Reads Between the Lines</div>
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error("⚠️ model.pkl ya vectorizer.pkl nahi mile. Pehle 'python train_model.py' chalao.")
    st.stop()

# ---------- Hero section ----------
col1, col2 = st.columns([1.3, 1])
with col1:
    st.markdown("""
    <div class="badge">⚡ Powered by Machine Learning</div>
    <div class="hero-title">Detect <span>Fake News</span><br>in Seconds</div>
    <div class="hero-sub">
        Paste any news article or headline below. Our AI model analyzes the text
        using TF-IDF and Logistic Regression to predict whether it's Real or Fake —
        instantly, with a confidence score.
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="ai-card">
        <div class="ai-icon">🤖</div>
        <h3>Smart Detection</h3>
        <p>Trained on thousands of real and fake news samples to spot misleading patterns.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------- Detector section ----------
st.markdown('<div class="section-title"><h2>Try the Detector</h2><p>Paste a news article and check instantly</p></div>', unsafe_allow_html=True)

if "news_input" not in st.session_state:
    st.session_state.news_input = ""

ecol1, ecol2 = st.columns(2)
example_fake = "SHOCKING you wont believe what vaccines secretly do to your body doctors hate this trick"
example_real = "The Reserve Bank of India announced new guidelines on digital payments after a meeting with banking officials"
with ecol1:
    if st.button("😱 Try Fake Example"):
        st.session_state.news_input = example_fake
with ecol2:
    if st.button("📰 Try Real Example"):
        st.session_state.news_input = example_real

news_text = st.text_area(
    "",
    height=170,
    placeholder="Paste news article text here...",
    value=st.session_state.news_input,
    key="news_input_box",
    label_visibility="collapsed"
)

check_clicked = st.button("🔍 Detect Now")

if check_clicked:
    if news_text.strip() == "":
        st.warning("Pehle kuch news text likho ya paste karo.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(news_text)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]
            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec)[0]
                confidence = max(proba) * 100
            elif hasattr(model, "decision_function"):
                # Models like Linear SVC have no predict_proba; squash the
                # decision score into a 0-100 "confidence" using a sigmoid.
                import math
                score = model.decision_function(vec)[0]
                confidence = (1 / (1 + math.exp(-abs(score)))) * 100

        result_class = "result-real" if prediction == "REAL" else "result-fake"
        result_label = "✅ REAL NEWS" if prediction == "REAL" else "🚫 FAKE NEWS"

        st.markdown(f"""
        <div class="result-card">
            <h3 class="{result_class}">{result_label}</h3>
            <p style="color:#64748b; margin-top:8px;">Confidence: <b>{confidence:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        if confidence is not None:
            st.progress(int(confidence))

st.markdown("<div style='margin-top:50px;'></div>", unsafe_allow_html=True)

# ---------- Feature cards ----------
st.markdown('<div class="section-title"><h2>How It Works</h2></div>', unsafe_allow_html=True)
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧹</div>
        <h4>Text Cleaning</h4>
        <p>Removes noise like URLs, punctuation and numbers before analysis.</p>
    </div>
    """, unsafe_allow_html=True)
with f2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔢</div>
        <h4>TF-IDF Vectorization</h4>
        <p>Converts text into numeric features weighted by word importance.</p>
    </div>
    """, unsafe_allow_html=True)
with f3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <h4>Logistic Regression</h4>
        <p>Classifies the article as Real or Fake with a confidence score.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Stats section ----------
try:
    df_stats = pd.read_csv("news_dataset.csv") if not (__import__("os").path.exists("True.csv")) else None
    total_rows = len(df_stats) if df_stats is not None else "44,000+"
except Exception:
    total_rows = "—"

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(f'<div class="stat-card"><h2>{total_rows}</h2><p>Training Samples</p></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="stat-card"><h2>2</h2><p>Classes (Real / Fake)</p></div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="stat-card"><h2>TF-IDF</h2><p>Feature Extraction</p></div>', unsafe_allow_html=True)
with s4:
    st.markdown('<div class="stat-card"><h2>Instant</h2><p>Prediction Speed</p></div>', unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("""
<div class="footer-box">
    Minor Project — Fake News Detection System | Built with <span>Streamlit + Scikit-learn</span>
</div>
""", unsafe_allow_html=True)
