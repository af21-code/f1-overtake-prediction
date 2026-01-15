import fastf1
import pandas as pd
import os

# 1. Configurazione della Cache
if not os.path.exists('../cache'):
    os.makedirs('../cache')

fastf1.Cache.enable_cache('../cache')

def download_data():
    # IL PERIODO "EFFETTO SUOLO"
    years = [2022, 2023, 2024]

    all_races_data = []

    print("🏎️ Inizio scaricamento dati 'Ground Effect Era' (2022-2024)...")

    for year in years:
        print(f"\n⬇️ Scarico dati: Gran Premio d'Italia {year}...")

        try:
            session = fastf1.get_session(year, 'Monza', 'R')
            session.load()

            laps = session.laps
            # Selezioniamo le colonne chiave
            cols = ['Driver', 'LapNumber', 'LapTime', 'Position', 'TyreLife', 'Compound']
            df = laps[cols].copy()

            # Aggiungiamo una colonna 'Year' per distinguere le gare (utile per debug)
            df['Year'] = year

            all_races_data.append(df)
            print(f"✅ Monza {year}: Caricati {len(df)} giri.")

        except Exception as e:
            print(f"⚠️ Errore con l'anno {year}: {e}")

    # 2. Unione dei dati
    if all_races_data:
        final_df = pd.concat(all_races_data)

        output_dir = '../data'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # CAMBIAMO IL NOME DEL FILE PER RIFLETTERE IL CONTENUTO
        output_path = f'{output_dir}/f1_ground_effect_dataset.csv'

        final_df.to_csv(output_path, index=False)

        print("\n" + "="*50)
        print(f"💾 DATASET COMPLETO SALVATO!")
        print(f"   Anni inclusi: {years}")
        print(f"   Totale giri: {len(final_df)}")
        print(f"   File: {output_path}")
        print("="*50)
    else:
        print("❌ Nessun dato scaricato.")

if __name__ == "__main__":
    download_data()