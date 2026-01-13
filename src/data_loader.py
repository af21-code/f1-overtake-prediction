import fastf1
import pandas as pd
import os

# 1. Configurazione della Cache
# Creiamo una cartella 'cache' per salvare i dati scaricati e non doverli riscaricare ogni volta
if not os.path.exists('../cache'):
    os.makedirs('../cache')

fastf1.Cache.enable_cache('../cache')

def test_data_download():
    print("🏎️ Inizio scaricamento dati GP Monza 2023...")

    # 2. Carichiamo la sessione di Gara ('R' = Race)
    # fastf1 gestisce tutto il download dall'API ufficiale
    session = fastf1.get_session(2023, 'Monza', 'R')
    session.load()

    # 3. Estraiamo i giri (Laps)
    laps = session.laps

    # Selezioniamo solo alcune colonne per vedere se funziona
    cols = ['Driver', 'LapNumber', 'LapTime', 'Position', 'TyreLife', 'Compound']
    df = laps[cols].copy()

    # 4. Mostriamo i risultati
    print(f"\n✅ Dati scaricati con successo! Trovati {len(df)} giri.")
    print("\nEcco le prime 5 righe del dataset:")
    print(df.head())

    # 5. Salviamo un file di prova nella cartella data
    output_path = '../data/test_monza.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 File salvato in: {output_path}")

if __name__ == "__main__":
    test_data_download()