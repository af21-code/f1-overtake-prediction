import streamlit as st
import pickle
import numpy as np
import os

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="F1 Strategy AI", page_icon="🏎️", layout="wide")

st.title("🏎️ F1 Overtake Prediction AI")
st.markdown("Previsione probabilità di sorpasso in tempo reale (Random Forest).")
st.write("---")

# --- CARICAMENTO RISORSE ---
@st.cache_resource
def load_resources():
    # Percorsi flessibili
    base_path = '../models' if os.path.exists('../models') else 'models'

    model_path = os.path.join(base_path, 'best_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    # Carichiamo anche l'encoder delle gomme per sicurezza
    encoder_path = os.path.join(base_path, 'le_compound.pkl')

    resources = {}

    try:
        with open(model_path, 'rb') as f:
            resources['model'] = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            resources['scaler'] = pickle.load(f)

        # Se l'encoder esiste lo carichiamo, altrimenti usiamo mappa manuale
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                resources['encoder'] = pickle.load(f)
        else:
            resources['encoder'] = None

        return resources
    except FileNotFoundError:
        st.error("❌ Errore: File necessari non trovati. Esegui il training prima!")
        return None

data = load_resources()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Telemetria")

# 1. NUOVO: NUMERO DEL GIRO (Necessario per lo scaler!)
lap_number = st.sidebar.slider("Giro della Gara", 1, 70, 10)

# 2. Posizione
position = st.sidebar.slider("Posizione Attuale", 1, 20, 8)

# 3. Usura Gomme
tyre_life = st.sidebar.slider("Usura Gomme (Giri percorsi)", 0, 40, 5)

# 4. Mescola
compound_label = st.sidebar.selectbox("Mescola", ["SOFT", "MEDIUM", "HARD"])

# Gestione Encoding Gomme
if data and data['encoder']:
    # Trasforma "SOFT" nel numero corretto usato nel training
    try:
        # Nota: l'encoder vuole un array 1D
        compound_val = data['encoder'].transform([compound_label])[0]
    except ValueError:
        # Fallback se la label non esiste (es. INTERMEDIATE non presenti nel train)
        compound_val = 0
else:
    # Mappa manuale di riserva (meno sicura ma funzionale)
    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
    compound_val = compound_map.get(compound_label, 0)

# 5. Tempo
lap_time = st.sidebar.number_input("Tempo sul Giro (sec)", 70.0, 120.0, 82.5)

if st.sidebar.button("🔮 Calcola Previsione"):
    if data:
        model = data['model']
        scaler = data['scaler']

        # ORDINE CRITICO: Deve essere identico a quello del training!
        # Nel feature_processor era: LapNumber, Position, TyreLife, Compound, LapTime_Sec
        # (Oppure l'ordine in cui le colonne erano nel dataframe X)

        # Costruiamo l'array con 5 VALORI
        raw_input = np.array([[lap_number, position, tyre_life, compound_val, lap_time]])

        # Scaling
        try:
            scaled_input = scaler.transform(raw_input)

            # Predizione
            # proba restituisce [prob_no_sorpasso, prob_sorpasso]
            probability = model.predict_proba(scaled_input)[0][1] * 100
            prediction = model.predict(scaled_input)[0]

            # Visualizzazione
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Esito Analisi:")
                if probability > 50:
                    st.success(f"🚀 ALTA PROBABILITÀ DI SORPASSO")
                    st.write("Il pilota ha il vantaggio prestazionale necessario.")
                else:
                    st.warning(f"🛡️ POSIZIONE MANTENUTA")
                    st.write("Non ci sono le condizioni per un attacco immediato.")

                st.progress(int(probability))

            with col2:
                st.metric(label="Chance Sorpasso", value=f"{probability:.1f}%")

        except ValueError as e:
            st.error(f"Errore di forma nei dati: {e}")
            st.info("Suggerimento: Controlla che il numero di input coincida con le colonne del training (5).")