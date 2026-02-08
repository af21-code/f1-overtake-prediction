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

# --- DRIVERS DATA 2024 ---
DRIVERS_2024 = [
    {"pos": 1, "code": "VER", "name": "Verstappen", "team": "Red Bull", "color": "#3671C6"},
    {"pos": 2, "code": "NOR", "name": "Norris", "team": "McLaren", "color": "#FF8000"},
    {"pos": 3, "code": "LEC", "name": "Leclerc", "team": "Ferrari", "color": "#E80020"},
    {"pos": 4, "code": "PIA", "name": "Piastri", "team": "McLaren", "color": "#FF8000"},
    {"pos": 5, "code": "SAI", "name": "Sainz", "team": "Ferrari", "color": "#E80020"},
    {"pos": 6, "code": "HAM", "name": "Hamilton", "team": "Mercedes", "color": "#27F4D2"},
    {"pos": 7, "code": "RUS", "name": "Russell", "team": "Mercedes", "color": "#27F4D2"},
    {"pos": 8, "code": "PER", "name": "Perez", "team": "Red Bull", "color": "#3671C6"},
    {"pos": 9, "code": "ALO", "name": "Alonso", "team": "Aston Martin", "color": "#229971"},
    {"pos": 10, "code": "STR", "name": "Stroll", "team": "Aston Martin", "color": "#229971"},
    {"pos": 11, "code": "HUL", "name": "Hulkenberg", "team": "Haas", "color": "#B6BABD"},
    {"pos": 12, "code": "TSU", "name": "Tsunoda", "team": "RB", "color": "#6692FF"},
    {"pos": 13, "code": "RIC", "name": "Ricciardo", "team": "RB", "color": "#6692FF"},
    {"pos": 14, "code": "ALB", "name": "Albon", "team": "Williams", "color": "#64C4FF"},
    {"pos": 15, "code": "GAS", "name": "Gasly", "team": "Alpine", "color": "#FF87BC"},
    {"pos": 16, "code": "OCO", "name": "Ocon", "team": "Alpine", "color": "#FF87BC"},
    {"pos": 17, "code": "MAG", "name": "Magnussen", "team": "Haas", "color": "#B6BABD"},
    {"pos": 18, "code": "BOT", "name": "Bottas", "team": "Sauber", "color": "#52E252"},
    {"pos": 19, "code": "ZHO", "name": "Zhou", "team": "Sauber", "color": "#52E252"},
    {"pos": 20, "code": "SAR", "name": "Sargeant", "team": "Williams", "color": "#64C4FF"},
]

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="F1 Pit Wall - Monza 2025",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD CSS ---
def load_css():
    """Carica gli stili CSS da file esterno."""
    css_path = os.path.join(os.path.dirname(__file__), 'style.css')
    with open(css_path, 'r', encoding='utf-8') as f:
        return f.read()

st.markdown(f'<style>{load_css()}</style>', unsafe_allow_html=True)


# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    base_path = '../../models' if os.path.exists('../../models') else 'models'
    try:
        with open(f'{base_path}/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{base_path}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return {'model': model, 'scaler': scaler}
    except FileNotFoundError:
        return None


def get_compound_value(compound):
    """
    Converte il tipo di mescola in un valore numerico per il calcolo del vantaggio.
    SOFT = 3 (massima aderenza), MEDIUM = 2, HARD = 1 (minima aderenza).
    """
    return {"SOFT": 3, "MEDIUM": 2, "HARD": 1}.get(compound, 0)


def predict_overtake(resources, attacker, defender):
    """
    Calcola la probabilità di sorpasso usando il modello ML addestrato.
    
    Le feature relative vengono calcolate come differenza tra attaccante e difensore:
    - Delta tempo sul giro (negativo = attaccante più veloce)
    - Delta usura gomme (negativo = gomme più fresche)
    - Vantaggio compound (positivo = mescola più performante)
    
    Returns:
        float: Probabilità percentuale di sorpasso (0-100)
    """
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
    st.markdown("""<div class="header-bar"><div><p class="header-title">MONZA 2025 — PIT WALL</p><p class="header-subtitle">Overtake Probability Analysis</p></div></div>""", unsafe_allow_html=True)
    
    if resources is None:
        st.error("Model not found. Run training pipeline first.")
        st.stop()
    
    # Initialize swap state
    if 'is_defense' not in st.session_state:
        st.session_state.is_defense = False
    
    # NEW LAYOUT: 2 main columns - Leaderboard LEFT, Prediction Tool RIGHT
    left_col, right_col = st.columns([1, 3])
    
    # ===== LEFT COLUMN: LEADERBOARD =====
    with left_col:
        gaps = ["Leader", "+2.3", "+4.1", "+5.8", "+8.2", "+12.5", "+14.1", "+18.3", 
                "+22.7", "+25.4", "+28.9", "+32.1", "+35.6", "+38.2", "+41.5", 
                "+44.8", "+48.3", "+52.1", "+56.7", "+61.2"]
        
        leaderboard_rows = ""
        for i, driver in enumerate(DRIVERS_2024):
            gap_text = gaps[i] if i < len(gaps) else f"+{i * 3:.1f}"
            leaderboard_rows += f'<div class="leaderboard-row"><span class="leaderboard-pos">{driver["pos"]}</span><span class="leaderboard-team-dot" style="background: {driver["color"]};"></span><span class="leaderboard-driver">{driver["code"]}</span><span class="leaderboard-gap">{gap_text}</span></div>'
        
        st.markdown(f'<div class="leaderboard-container" style="max-height: 500px;"><p class="leaderboard-title">Live Standings</p>{leaderboard_rows}</div>', unsafe_allow_html=True)
    
    # ===== RIGHT COLUMN: PREDICTION TOOL =====
    with right_col:
        # Sub-columns for driver inputs
        driver_col1, gap_col, driver_col2 = st.columns([3, 1.5, 3])
        
        # YOUR DRIVER
        with driver_col1:
            st.markdown('<p class="panel-title attacker">YOUR DRIVER</p>', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                my_position = st.slider("Position", 2, 20, 5, key="pos")
                my_compound = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"], index=1, key="comp")
            with c2:
                my_tyre_life = st.slider("Tyre Age (laps)", 1, 40, 12, key="tyre")
                lap_col1, lap_col2 = st.columns(2)
                with lap_col1:
                    my_lap_minutes = st.number_input("Minutes", 1, 2, 1, 1, key="my_min")
                with lap_col2:
                    my_lap_seconds = st.number_input("Seconds", 0.0, 59.9, 24.5, 0.1, key="my_sec")
                my_lap_time = my_lap_minutes * 60 + my_lap_seconds
        
        # GAP & CONTROLS
        with gap_col:
            st.markdown('<div style="text-align: center; padding: 20px 0;"><p style="color: #6e7681; font-size: 0.7em; text-transform: uppercase; letter-spacing: 2px; margin: 0;">Gap to Target</p></div>', unsafe_allow_html=True)
            
            gap = st.number_input("Gap (s)", 0.1, 10.0, 0.8, 0.1, key="gap", label_visibility="collapsed")
            
            st.markdown(f'<div style="text-align: center; margin: 10px 0 20px 0;"><span style="font-size: 2.2em; font-weight: 700; color: #f0f6fc; font-family: Monaco, monospace;">{gap:.1f}<span style="font-size: 0.5em; color: #6e7681;">s</span></span></div>', unsafe_allow_html=True)
            
            if st.button("SWAP", key="swap", use_container_width=True):
                st.session_state.is_defense = not st.session_state.is_defense
                st.rerun()
            
            mode_class = "defense" if st.session_state.is_defense else "attack"
            mode_text = "DEFENSE" if st.session_state.is_defense else "ATTACK"
            st.markdown(f'<div style="text-align:center"><span class="mode-indicator {mode_class}">{mode_text}</span></div>', unsafe_allow_html=True)
        
        # OPPONENT
        with driver_col2:
            st.markdown(f'<p class="panel-title defender">OPPONENT — P{my_position - 1}</p>', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                opp_compound = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"], index=2, key="opp_comp")
                opp_tyre_life = st.slider("Tyre Age (laps)", 1, 40, 25, key="opp_tyre")
            with c2:
                opp_lap_col1, opp_lap_col2 = st.columns(2)
                with opp_lap_col1:
                    opp_lap_minutes = st.number_input("Minutes", 1, 2, 1, 1, key="opp_min")
                with opp_lap_col2:
                    opp_lap_seconds = st.number_input("Seconds", 0.0, 59.9, 25.2, 0.1, key="opp_sec")
                opp_lap_time = opp_lap_minutes * 60 + opp_lap_seconds
        
        # Calculate button
        st.markdown("---")
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
            
            if prob > 60:
                level, label = "high", "HIGH PROBABILITY"
            elif prob > 40:
                level, label = "medium", "UNCERTAIN"
            else:
                level, label = "low", "LOW PROBABILITY"
            
            _, res_col, _ = st.columns([1, 3, 1])
            with res_col:
                st.markdown(f'<div class="result-panel {level}"><p class="result-percentage {level}">{prob:.1f}%</p><p class="result-label">{label} {context}</p></div>', unsafe_allow_html=True)
                st.progress(int(min(prob, 100)))
                
                delta = attacker['lap_time'] - defender['lap_time']
                delta_sign = "faster" if delta < 0 else "slower"
                st.caption(f"Delta: {abs(delta):.2f}s {delta_sign} | Tyre diff: {attacker['tyre_life'] - defender['tyre_life']:+d} laps")
    
    # ===== BOTTOM: MONZA CIRCUIT =====
    st.markdown("---")
    
    driver_dots = ""
    for i, driver in enumerate(DRIVERS_2024):
        driver_dots += f'<circle class="driver-dot driver-dot-{i+1}" r="6" fill="{driver["color"]}" style="color: {driver["color"]};"><title>{driver["code"]} - {driver["name"]}</title></circle>'
    
    track_path = "M 50,180 L 50,60 C 50,30 80,15 120,15 L 350,15 C 390,15 420,30 420,60 L 420,100 C 420,120 400,140 370,140 L 280,140 C 250,140 230,160 230,180 L 230,200 C 230,220 200,240 170,240 L 50,240 L 50,180"
    
    circuit_html = f'<div class="circuit-container"><p class="circuit-title">Autodromo Nazionale Monza — Live Track</p><svg class="circuit-svg" viewBox="0 0 500 280" xmlns="http://www.w3.org/2000/svg"><path d="{track_path}" fill="none" stroke="#1a1a2e" stroke-width="28" stroke-linecap="round" stroke-linejoin="round"/><path d="{track_path}" fill="none" stroke="#2d3748" stroke-width="18" stroke-linecap="round" stroke-linejoin="round"/><path d="{track_path}" fill="none" stroke="#4a5568" stroke-width="1" stroke-dasharray="8,8"/><rect x="45" y="200" width="10" height="3" fill="#ffffff"/><rect x="45" y="203" width="10" height="3" fill="#e10600"/><text x="120" y="8" class="corner-label">Variante del Rettifilo</text><text x="380" y="50" class="corner-label">Curva Grande</text><text x="430" y="90" class="corner-label">Lesmo 1</text><text x="380" y="155" class="corner-label">Lesmo 2</text><text x="280" y="180" class="corner-label">Ascari</text><text x="120" y="260" class="corner-label">Parabolica</text>{driver_dots}</svg></div>'
    
    st.markdown(circuit_html, unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer-text">F1 Overtake Prediction • Model trained on Monza 2022-2024 data • ML Course Project</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()