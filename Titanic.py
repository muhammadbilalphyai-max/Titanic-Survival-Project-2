import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0a0a2e 0%, #1a1a4e 50%, #0d2137 100%);
    padding: 15px 30px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}
.hero h1 { font-size: 1.8rem; font-weight: 700; color: white; margin: 0; }
.hero p { color: #a0aec0; font-size: 0.85rem; margin: 5px 0 0 0; }

.metric-card {
    background: linear-gradient(135deg, #1e2749 0%, #16213e 100%);
    padding: 10px;
    border-radius: 10px;
    border: 1px solid rgba(99, 179, 237, 0.2);
    text-align: center;
    margin-bottom: 8px;
}
.metric-icon { font-size: 1.2rem; margin-bottom: 2px; }
.metric-label { color: #a0aec0; font-size: 0.65rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
.metric-value { color: white; font-size: 1rem; font-weight: 700; margin-top: 2px; }

.result-survived {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    margin-top: 10px;
}
.result-died {
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    margin-top: 10px;
}
.result-text { font-size: 2rem; font-weight: 700; color: white; }
.result-label { color: rgba(255,255,255,0.8); font-size: 0.85rem; margin: 0; }

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 10px 30px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a2e 0%, #1a1a4e 100%);
    border-right: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: white !important;
}
.sidebar-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #63b3ed !important;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

/* Center button */
div[data-testid="stButton"] {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("LogisticTitanicModel.joblib")

# Hero
st.markdown("""
<div class="hero">
    <h1>🚢 Titanic Survival Predictor</h1>
    <p>Would you have survived the Titanic? Find out using Machine Learning!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<div class="sidebar-title">👤 Passenger Details</div>', unsafe_allow_html=True)

Passenger_Class_Label = st.sidebar.selectbox(
    "🎫 Passenger Class",
    ["1st Class", "2nd Class", "3rd Class"],
    index=2
)
Passenger_Class = {"1st Class": 1, "2nd Class": 2, "3rd Class": 3}[Passenger_Class_Label]

Gender = st.sidebar.selectbox("👤 Gender", ["Male", "Female"])
gender_code = 1 if Gender == "Male" else 0

Age = st.sidebar.slider("🎂 Age", 1, 80, 28)
Siblings_Spouses = st.sidebar.slider("👫 Siblings/Spouses", 0, 8, 0)
Parents_Children = st.sidebar.slider("👨‍👧 Parents/Children", 0, 6, 0)
Ticket_Fare = st.sidebar.slider("💰 Ticket Fare ($)", 0, 520, 14)

Embarkation_Port = st.sidebar.selectbox(
    "⚓ Embarkation Port",
    ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"]
)
port_code = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}[Embarkation_Port]

# Summary cards
st.markdown("##### 📊 Passenger Summary")
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

cards = [
    (col1, "🎫", "Class", Passenger_Class_Label),
    (col2, "👤", "Gender", Gender),
    (col3, "🎂", "Age", Age),
    (col4, "👫", "Siblings", Siblings_Spouses),
    (col5, "👨‍👧", "Parents", Parents_Children),
    (col6, "💰", "Fare", f"${Ticket_Fare}"),
    (col7, "⚓", "Port", Embarkation_Port.split()[0]),
]

for col, icon, label, value in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Centered predict button
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    predict = st.button("🔍 Predict")

if predict:
    with st.spinner("🤖 Analyzing..."):
        pred = model.predict([[
            Passenger_Class, gender_code, Age,
            Siblings_Spouses, Parents_Children,
            Ticket_Fare, port_code
        ]])
        result = pred[0]

    if result == 1:
        st.markdown("""
        <div class="result-survived">
            <div class="result-label">Prediction Result</div>
            <div class="result-text">✅ SURVIVED!</div>
            <div class="result-label">You would have survived the Titanic!</div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("""
        <div class="result-died">
            <div class="result-label">Prediction Result</div>
            <div class="result-text">❌ DID NOT SURVIVE</div>
            <div class="result-label">Unfortunately you would not have survived.</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align:center; color:#4a5568; padding:10px; font-size:0.75rem;'>
    🚢 Titanic Survival Predictor | Powered by Machine Learning | Educational Purposes Only
</div>
""", unsafe_allow_html=True)