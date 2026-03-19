import streamlit as st
from streamlit_lottie import st_lottie
import requests
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="🌟 Fitness AI App",
    page_icon="💪",
    layout="wide"
)

# -----------------------------
# Load Lottie Animation (background)
# -----------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_bg = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
st_lottie(lottie_bg, height=350, key="bg")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
    <div style='text-align: center; color: white;'>
        <h1>Fitness & Health Analyzer</h1>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# User Inputs (Number Inputs for mobile-friendly)
# -----------------------------
st.header("Enter Your Data")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=25, step=1)
    weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=70)
    height = st.number_input("Height (cm)", min_value=140, max_value=210, value=170)

with col2:
    sleep = st.number_input("Sleep (hours)", min_value=0, max_value=12, value=7)
    exercise = st.number_input("Exercise (hours/week)", min_value=0, max_value=14, value=3)
    steps = st.number_input("Daily Steps", min_value=0, max_value=20000, value=6000)
    calories = st.number_input("Calories", min_value=1000, max_value=4000, value=2000)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Analyze"):
    with st.spinner("Analyzing your health..."):
        time.sleep(2)

    # Prepare Data
    user_data = np.array([[age, weight, height, sleep, exercise, steps, calories]])
    user_scaled = scaler.transform(user_data)
    cluster = model.predict(user_scaled)[0]

    # Mapping clusters
    cluster_names = {
        0: ("⚖️ Average Lifestyle", "warning"),
        1: ("🛑 Sedentary", "error"),
        2: ("💪 Active & Healthy", "success")
    }

    msg, level = cluster_names[cluster]
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    else:
        st.error(msg)

    # Radar Chart
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[sleep, exercise, steps/2000, calories/1000],
        theta=['Sleep','Exercise','Steps','Calories'],
        fill='toself',
        name="Your Profile"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(12,14,10,4)])),
        showlegend=False,
        title="Your Health Profile"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")