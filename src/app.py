"""
================================================================================
APP.PY - F1 Overtake Prediction Web Application
================================================================================
Webapp Streamlit che simula un muretto box durante il GP di Monza 2025.
Permette di confrontare due piloti e calcolare la probabilita di sorpasso
basandosi su un modello di Machine Learning addestrato su dati storici.

Funzionalita:
- Input parametri per il proprio pilota (posizione, gomme, usura, tempo)
- Input parametri per l'avversario
- Calcolo probabilita di sorpasso basato su feature relative
- Modalita ATTACK (probabilita di sorpassare) e DEFENSE (di essere sorpassato)
- Visualizzazione risultato con barra di progresso

Architettura:
- Caricamento modello pre-addestrato (best_model.pkl)
- Caricamento scaler per normalizzazione (scaler.pkl)
- Calcolo feature relative tra i due piloti
- Predizione probabilita con predict_proba()

Modello utilizzato: Logistic Regression con class_weight='balanced'
Dati training: Monza GP 2022, 2023, 2024
================================================================================
"""

import streamlit as st
import pickle
import numpy as np
import os
import json

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="F1 Pit Wall - Monza 2025",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PROFESSIONAL CSS ---
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark F1 theme */
    .stApp {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    
    /* Compact header */
    .header-bar {
        background: linear-gradient(90deg, #e10600 0%, #b30500 100%);
        padding: 12px 24px;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .header-title {
        color: white;
        font-size: 1.4em;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.7);
        font-size: 0.85em;
        margin: 0;
    }
    
    /* Driver panels */
    .driver-panel {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 16px;
    }
    
    .driver-panel.attacker {
        border-left: 3px solid #00d26a;
    }
    
    .driver-panel.defender {
        border-left: 3px solid #ff4757;
    }
    
    .panel-title {
        color: #8b949e;
        font-size: 0.75em;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        font-weight: 600;
    }
    
    .panel-title.attacker { color: #00d26a; }
    .panel-title.defender { color: #ff4757; }
    
    /* Gap display */
    .gap-container {
        background: rgba(255,215,0,0.08);
        border: 1px solid rgba(255,215,0,0.2);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    
    .gap-label {
        color: #8b949e;
        font-size: 0.7em;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    
    .gap-value {
        color: #ffd700;
        font-size: 1.8em;
        font-weight: 700;
        font-family: 'Monaco', 'Consolas', monospace;
    }
    
    /* Result panel */
    .result-panel {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 16px;
    }
    
    .result-panel.high {
        border-color: #00d26a;
        background: rgba(0,210,106,0.05);
    }
    
    .result-panel.medium {
        border-color: #ffd700;
        background: rgba(255,215,0,0.05);
    }
    
    .result-panel.low {
        border-color: #ff4757;
        background: rgba(255,71,87,0.05);
    }
    
    .result-percentage {
        font-size: 3em;
        font-weight: 700;
        font-family: 'Monaco', 'Consolas', monospace;
        margin: 0;
    }
    
    .result-percentage.high { color: #00d26a; }
    .result-percentage.medium { color: #ffd700; }
    .result-percentage.low { color: #ff4757; }
    
    .result-label {
        color: #8b949e;
        font-size: 0.85em;
        margin-top: 4px;
    }
    
    /* Buttons */
    .stButton > button {
        background: #e10600;
        color: white;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #ff1a1a;
        box-shadow: 0 4px 12px rgba(225,6,0,0.3);
    }
    
    /* Mode toggle */
    .mode-indicator {
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.75em;
        color: #8b949e;
        display: inline-block;
        margin-top: 8px;
    }
    
    .mode-indicator.attack { color: #00d26a; border: 1px solid #00d26a; }
    .mode-indicator.defense { color: #ff4757; border: 1px solid #ff4757; }
    
    /* Inputs styling */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #c9d1d9 !important;
        font-size: 0.85em !important;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.4em;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #e10600, #ff4757);
    }
    
    /* Footer */
    .footer-text {
        color: #484f58;
        font-size: 0.7em;
        text-align: center;
        padding: 10px;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)


# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    base_path = '../models' if os.path.exists('../models') else 'models'
    try:
        with open(f'{base_path}/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{base_path}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return {'model': model, 'scaler': scaler}
    except FileNotFoundError:
        return None


def get_compound_value(compound):
    return {"SOFT": 3, "MEDIUM": 2, "HARD": 1}.get(compound, 0)


def predict_overtake(resources, attacker, defender):
    model, scaler = resources['model'], resources['scaler']
    
    features = np.array([[
        attacker['position'],
        attacker['lap_time'] - defender['lap_time'],
        attacker['tyre_life'] - defender['tyre_life'],
        get_compound_value(attacker['compound']) - get_compound_value(defender['compound']),
        attacker['lap_time'],
        attacker['tyre_life']
    ]])
    
    features_scaled = scaler.transform(features)
    return model.predict_proba(features_scaled)[0][1] * 100


# --- MAIN APP ---
def main():
    resources = load_resources()
    
    # Header
    st.markdown("""
    <div class="header-bar">
        <div>
            <p class="header-title">MONZA 2025 — PIT WALL</p>
            <p class="header-subtitle">Overtake Probability Analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if resources is None:
        st.error("Model not found. Run training pipeline first.")
        st.stop()
    
    # Initialize swap state
    if 'is_defense' not in st.session_state:
        st.session_state.is_defense = False
    
    # Main layout: 3 columns
    col1, col2, col3 = st.columns([4, 2, 4])
    
    # LEFT: YOUR DRIVER
    with col1:
        st.markdown(f'<p class="panel-title attacker">YOUR DRIVER</p>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            my_position = st.slider("Position", 2, 20, 5, key="pos")
            my_compound = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"], index=1, key="comp")
        with c2:
            my_tyre_life = st.slider("Tyre Age (laps)", 1, 40, 12, key="tyre")
            my_lap_time = st.number_input("Lap Time (s)", 80.0, 95.0, 84.5, 0.1, key="lap")
    
    # CENTER: GAP & CONTROLS
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <p style="color: #6e7681; font-size: 0.7em; text-transform: uppercase; letter-spacing: 2px; margin: 0;">Gap to Target</p>
        </div>
        """, unsafe_allow_html=True)
        
        gap = st.number_input("Gap (s)", 0.1, 10.0, 0.8, 0.1, key="gap", label_visibility="collapsed")
        
        st.markdown(f"""
        <div style="text-align: center; margin: 10px 0 20px 0;">
            <span style="font-size: 2.2em; font-weight: 700; color: #f0f6fc; font-family: 'Monaco', monospace;">{gap:.1f}<span style="font-size: 0.5em; color: #6e7681;">s</span></span>
        </div>
        """, unsafe_allow_html=True)
        
        # Swap button
        if st.button("SWAP", key="swap", use_container_width=True):
            st.session_state.is_defense = not st.session_state.is_defense
            st.rerun()
        
        mode_class = "defense" if st.session_state.is_defense else "attack"
        mode_text = "DEFENSE" if st.session_state.is_defense else "ATTACK"
        st.markdown(f'<div style="text-align:center"><span class="mode-indicator {mode_class}">{mode_text}</span></div>', unsafe_allow_html=True)
    
    # RIGHT: OPPONENT
    with col3:
        st.markdown(f'<p class="panel-title defender">OPPONENT — P{my_position - 1}</p>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            opp_compound = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"], index=2, key="opp_comp")
            opp_tyre_life = st.slider("Tyre Age (laps)", 1, 40, 25, key="opp_tyre")
        with c2:
            opp_lap_time = st.number_input("Lap Time (s)", 80.0, 95.0, 85.2, 0.1, key="opp_lap")
    
    st.markdown("---")
    
    # Calculate button centered
    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        calculate = st.button("CALCULATE PROBABILITY", use_container_width=True)
    
    # Results
    if calculate:
        if st.session_state.is_defense:
            attacker = {'position': my_position - 1, 'compound': opp_compound, 
                       'tyre_life': opp_tyre_life, 'lap_time': opp_lap_time}
            defender = {'compound': my_compound, 'tyre_life': my_tyre_life, 'lap_time': my_lap_time}
            context = "of being overtaken"
        else:
            attacker = {'position': my_position, 'compound': my_compound,
                       'tyre_life': my_tyre_life, 'lap_time': my_lap_time}
            defender = {'compound': opp_compound, 'tyre_life': opp_tyre_life, 'lap_time': opp_lap_time}
            context = "of overtaking"
        
        prob = predict_overtake(resources, attacker, defender)
        
        # Determine styling
        if prob > 60:
            level, label = "high", "HIGH PROBABILITY"
        elif prob > 40:
            level, label = "medium", "UNCERTAIN"
        else:
            level, label = "low", "LOW PROBABILITY"
        
        # Result display
        _, res_col, _ = st.columns([1, 3, 1])
        with res_col:
            st.markdown(f"""
            <div class="result-panel {level}">
                <p class="result-percentage {level}">{prob:.1f}%</p>
                <p class="result-label">{label} {context}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(int(min(prob, 100)))
            
            # Compact analysis
            delta = attacker['lap_time'] - defender['lap_time']
            delta_sign = "faster" if delta < 0 else "slower"
            st.caption(f"Delta: {abs(delta):.2f}s {delta_sign} | Tyre diff: {attacker['tyre_life'] - defender['tyre_life']:+d} laps")
    
    # Footer
    st.markdown("""
    <div class="footer-text">
        F1 Overtake Prediction • Model trained on Monza 2022-2024 data • ML Course Project
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()