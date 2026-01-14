import fastf1
import pandas as pd
import numpy as np
import os

# 1. SETUP INIZIALE
# Creiamo la cache per non scaricare i dati mille volte
if not os.path.exists('../cache'):
    os.makedirs('../cache')
fastf1.Cache.enable_cache('../cache')

def process_race(year, gp_name):
    """
    Scarica una gara, calcola i sorpassi e pulisce i dati dai Pit Stop.
    """
    print(f"\n🔄 Elaborazione {gp_name} {year}...")

    try:
        # Carichiamo la sessione di Gara ('R')
        # telemetry=True serve per dati precisi, ma è pesante. Per ora usiamo i dati lap-by-lap.
        session = fastf1.get_session(year, gp_name, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps
    except Exception as e:
        print(f"⚠️ Saltato {gp_name}: {e}")
        return None

    # 2. SELEZIONE FEATURE (Le colonne che ci servono)
    # Driver: Pilota
    # LapNumber: Numero del giro
    # Position: Posizione attuale
    # LapTime: Tempo sul giro
    # TyreLife: Usura gomme (giri percorsi con quel set)
    # Compound: Tipo di gomma (Soft, Medium, Hard)
    cols = ['Driver', 'LapNumber', 'LapTime', 'Position', 'TyreLife', 'Compound', 'Team']

    # Filtriamo solo le colonne che esistono davvero
    actual_cols = [c for c in cols if c in laps.columns]
    df = laps[actual_cols].copy()

    # Pulizia base: Via i giri senza tempo o posizione
    df.dropna(subset=['Position', 'LapTime'], inplace=True)

    # 3. CALCOLO DEL TARGET (Sorpasso Sì/No)
    # Ordiniamo per Pilota e Giro per poter confrontare il giro X con X+1
    df = df.sort_values(by=['Driver', 'LapNumber'])

    # Creiamo la colonna 'NextPosition' (Posizione al giro dopo)
    df['NextPosition'] = df.groupby('Driver')['Position'].shift(-1)

    # Logica: Se la posizione attuale è PEGGIORE (numero più alto) della prossima
    # Esempio: Ero 5°, divento 4° -> 5 > 4 -> True (Sorpasso)
    df['IsOvertake'] = (df['Position'] > df['NextPosition']).astype(int)

    # Rimuoviamo l'ultimo giro di ogni pilota (non ha un "futuro" da predire)
    df.dropna(subset=['NextPosition'], inplace=True)

    # 4. GESTIONE ISSUE: I PIT STOP [Punto cruciale per il Report]
    # Un sorpasso ai box non è merito del pilota ma della strategia. Vogliamo predire sorpassi in pista.
    # Euristica: Se il tempo sul giro è molto alto (> 115% della media), è un pit stop o Safety Car.

    # Convertiamo LapTime in secondi
    df['LapTime_Sec'] = df['LapTime'].dt.total_seconds()
    avg_lap = df['LapTime_Sec'].mean()

    # Identifichiamo i giri anomali (Lenti)
    df['IsSlowLap'] = df['LapTime_Sec'] > (avg_lap * 1.15)

    # TENIAMO SOLO I GIRI "NORMALI"
    # Scartiamo i giri dove il pilota era lento (box/incidenti)
    df_clean = df[df['IsSlowLap'] == False].copy()

    # Aggiungiamo il nome della gara per riferimento
    df_clean['GP_Name'] = gp_name

    print(f"✅ {gp_name}: {len(df_clean)} giri validi | {df_clean['IsOvertake'].sum()} sorpassi rilevati")
    return df_clean

def build_season_dataset(year):
    print(f"🚀 Inizio scaricamento stagione {year}...")

    # Otteniamo il calendario delle gare
    schedule = fastf1.get_event_schedule(year)

    # Filtriamo solo le gare convenzionali già disputate
    # Prendiamo solo le prime 5 gare per testare velocemente il codice (poi toglieremo il limite)
    races = schedule[schedule['EventFormat'] == 'conventional']['EventName'].unique()

    # --- MODIFICA QUI PER SCARICARE TUTTO O SOLO UN PO' ---
    races_to_download = races # Scarica TUTTO L'ANNO
    # races_to_download = races   # Se vuoi scaricare TUTTO l'anno (ci mette 5-10 minuti)

    all_data = []

    for race_name in races_to_download:
        race_df = process_race(year, race_name)
        if race_df is not None:
            all_data.append(race_df)

    # Uniamo tutto in un unico grande file
    if len(all_data) > 0:
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Costruiamo il dataset del 2023
    dataset = build_season_dataset(2023)

    if not dataset.empty:
        # Creiamo cartella data se non esiste
        if not os.path.exists('../data'):
            os.makedirs('../data')

        output_file = '../data/f1_2023_processed.csv'
        dataset.to_csv(output_file, index=False)

        print("\n" + "="*50)
        print(f"💾 DATASET SALVATO: {output_file}")
        print(f"📊 Dimensioni totali: {dataset.shape}")

        # Statistiche per il Report (Dati Sbilanciati)
        n_overtakes = dataset['IsOvertake'].sum()
        perc = (n_overtakes / len(dataset)) * 100
        print(f"📉 Totale Sorpassi: {n_overtakes}")
        print(f"⚠️ Sbilanciamento Classi: Solo il {perc:.2f}% dei giri ha un sorpasso.")
        print("="*50)
    else:
        print("❌ Nessun dato scaricato.")