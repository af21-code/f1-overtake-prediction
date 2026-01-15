import fastf1
import os

# 1. Configurazione della Cache
if not os.path.exists('../cache'):
    os.makedirs('../cache')

fastf1.Cache.enable_cache('../cache')

def test_data_download():
    print("🏎️ Inizio scaricamento dati GP Monza 2023...")

    # 2. Carichiamo la sessione
    session = fastf1.get_session(2023, 'Monza', 'R')
    session.load()

    # 3. Estraiamo i giri
    laps = session.laps
    cols = ['Driver', 'LapNumber', 'LapTime', 'Position', 'TyreLife', 'Compound']
    df = laps[cols].copy()

    # 4. Mostriamo i risultati
    print(f"\n✅ Dati scaricati con successo! Trovati {len(df)} giri.")
    print("\nEcco le prime 5 righe del dataset:")
    print(df.head())

    # --- FIX CORRETTO (Tutto indentato qui dentro) ---
    output_dir = '../data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Cartella '{output_dir}' creata.")

    output_path = f'{output_dir}/f1_2023_processed.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dati salvati in: {output_path}")

if __name__ == "__main__":
    test_data_download()