"""
================================================================================
DATA LOADER - F1 Overtake Prediction
================================================================================
Questo script scarica i dati delle gare di Formula 1 dal GP di Monza usando 
la libreria FastF1. Vengono estratti i dati lap-by-lap per le stagioni 
2022, 2023 e 2024 (era "Ground Effect").

Dati estratti per ogni giro:
- Driver: codice pilota (es. VER, HAM, LEC)
- LapNumber: numero del giro
- LapTime: tempo sul giro
- Position: posizione in classifica
- TyreLife: numero di giri con il set di gomme attuale
- Compound: tipo di mescola (SOFT, MEDIUM, HARD)

Output: data/f1_ground_effect_dataset.csv
================================================================================
"""

import fastf1
import pandas as pd
import os

# Configurazione della Cache FastF1
if not os.path.exists('../cache'):
    os.makedirs('../cache')

fastf1.Cache.enable_cache('../cache')


def download_data():
    """
    Scarica i dati delle gare di Monza per gli anni 2022-2024.
    Salva il dataset combinato in formato CSV.
    """
    years = [2022, 2023, 2024]
    all_races_data = []

    print("=" * 60)
    print("F1 DATA LOADER - Monza Grand Prix (2022-2024)")
    print("=" * 60)

    for year in years:
        print(f"\n[INFO] Scaricamento dati: Gran Premio d'Italia {year}...")

        try:
            session = fastf1.get_session(year, 'Monza', 'R')
            session.load()

            laps = session.laps
            cols = ['Driver', 'LapNumber', 'LapTime', 'Position', 'TyreLife', 'Compound']
            df = laps[cols].copy()

            # Aggiungiamo colonna Year per distinguere le gare
            df['Year'] = year

            all_races_data.append(df)
            print(f"[OK] Monza {year}: Caricati {len(df)} giri.")

        except Exception as e:
            print(f"[ERRORE] Anno {year}: {e}")

    # Unione dei dati
    if all_races_data:
        final_df = pd.concat(all_races_data)

        output_dir = '../data'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = f'{output_dir}/f1_ground_effect_dataset.csv'
        final_df.to_csv(output_path, index=False)

        print("\n" + "=" * 60)
        print("DATASET SALVATO CON SUCCESSO")
        print(f"  Anni inclusi: {years}")
        print(f"  Totale giri: {len(final_df)}")
        print(f"  File: {output_path}")
        print("=" * 60)
    else:
        print("[ERRORE] Nessun dato scaricato.")


if __name__ == "__main__":
    download_data()