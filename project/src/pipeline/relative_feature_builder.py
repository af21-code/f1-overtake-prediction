"""
================================================================================
RELATIVE FEATURE BUILDER - F1 Overtake Prediction
================================================================================
Questo script costruisce le feature RELATIVE tra coppie di piloti adiacenti
in pista. Queste feature sono fondamentali per predire la probabilità di
un sorpasso, poiché catturano la dinamica del duello tra attaccante e difensore.

Feature generate:
- Delta_LapTime: differenza tempo sul giro (negativo = attaccante piu veloce)
- Delta_TyreLife: differenza usura gomme (negativo = gomme piu fresche)
- Compound_Advantage: vantaggio tipo gomma (SOFT=3, MEDIUM=2, HARD=1)
- Estimated_Gap: gap stimato tra i piloti
- Attacker_Position: posizione dell'attaccante
- Attacker_LapTime: tempo sul giro dell'attaccante
- Attacker_TyreLife: usura gomme dell'attaccante

Target:
- IsOvertake: 1 se l'attaccante ha guadagnato posizione nel giro successivo

Input: data/f1_ground_effect_dataset.csv
Output: data/f1_monza_relative_features.csv
================================================================================
"""

import pandas as pd
import numpy as np
import os

# Mappa per l'ordine delle gomme (vantaggi prestazionali)
COMPOUND_ORDER = {'SOFT': 3, 'MEDIUM': 2, 'HARD': 1, 'INTERMEDIATE': 0, 'WET': 0}


def load_raw_data():
    """Carica il dataset grezzo di Monza."""
    data_path = '../../data/f1_ground_effect_dataset.csv'
    
    if not os.path.exists(data_path):
        print("[ERRORE] Dataset non trovato. Esegui prima data_loader.py!")
        return None
    
    df = pd.read_csv(data_path)
    print(f"[OK] Dataset caricato: {df.shape}")
    print(f"     Anni: {df['Year'].unique().tolist()}")
    
    return df


def convert_laptime_to_seconds(df):
    """Converte LapTime in secondi."""
    if df['LapTime'].dtype == object:
        df['LapTime'] = df['LapTime'].astype(str).str.replace('0 days ', '', regex=False)
        df['LapTime_Sec'] = pd.to_timedelta(df['LapTime'], errors='coerce').dt.total_seconds()
    else:
        df['LapTime_Sec'] = df['LapTime']
    
    return df


def build_relative_features(df):
    """
    Costruisce le feature relative tra piloti adiacenti.
    Per ogni giro, confrontiamo ogni pilota con quello davanti.
    """
    print("\n[INFO] Costruzione feature relative...")
    
    # 1. Convertiamo LapTime
    df = convert_laptime_to_seconds(df)
    
    # 2. Rimuoviamo giri con dati mancanti
    df = df.dropna(subset=['Position', 'LapTime_Sec', 'TyreLife', 'Compound'])
    
    # Filtro outlier: rimozione giri con tempo > media + 2σ
    # Elimina pit stop, safety car, giri di formazione e anomalie telemetriche
    avg_lap = df['LapTime_Sec'].mean()
    std_lap = df['LapTime_Sec'].std()
    df = df[df['LapTime_Sec'] < avg_lap + 2 * std_lap]
    
    # 4. Ordiniamo per costruire le coppie
    df = df.sort_values(by=['Year', 'LapNumber', 'Position'])
    
    # 5. Per ogni pilota, troviamo i dati del pilota DAVANTI
    relative_data = []
    
    for (year, lap_num), lap_group in df.groupby(['Year', 'LapNumber']):
        lap_sorted = lap_group.sort_values('Position')
        
        for i in range(1, len(lap_sorted)):
            # Pilota che potrebbe sorpassare (dietro)
            attacker = lap_sorted.iloc[i]
            # Pilota davanti
            defender = lap_sorted.iloc[i-1]
            
            # Verifica che siano posizioni adiacenti
            if attacker['Position'] - defender['Position'] != 1:
                continue
            
            # Calcolo feature relative
            delta_laptime = attacker['LapTime_Sec'] - defender['LapTime_Sec']
            delta_tyre_life = attacker['TyreLife'] - defender['TyreLife']
            
            attacker_compound = COMPOUND_ORDER.get(attacker['Compound'], 0)
            defender_compound = COMPOUND_ORDER.get(defender['Compound'], 0)
            compound_advantage = attacker_compound - defender_compound
            
            # Gap stimato: approssimazione basata sulla variazione del distacco
            # Fattore 0.5 modella la progressiva chiusura/apertura del gap per giro
            estimated_gap = abs(delta_laptime) * 0.5
            
            # Target: l'attacker ha guadagnato posizione nel giro successivo?
            next_lap = df[(df['Year'] == year) & 
                          (df['LapNumber'] == lap_num + 1) & 
                          (df['Driver'] == attacker['Driver'])]
            
            if len(next_lap) == 0:
                continue
            
            next_position = next_lap.iloc[0]['Position']
            overtake_happened = 1 if next_position < attacker['Position'] else 0
            
            relative_data.append({
                'Year': year,
                'LapNumber': lap_num,
                'Attacker_Driver': attacker['Driver'],
                'Defender_Driver': defender['Driver'],
                'Attacker_Position': attacker['Position'],
                'Attacker_LapTime': attacker['LapTime_Sec'],
                'Attacker_TyreLife': attacker['TyreLife'],
                'Attacker_Compound': attacker['Compound'],
                'Defender_LapTime': defender['LapTime_Sec'],
                'Defender_TyreLife': defender['TyreLife'],
                'Defender_Compound': defender['Compound'],
                'Delta_LapTime': delta_laptime,
                'Delta_TyreLife': delta_tyre_life,
                'Compound_Advantage': compound_advantage,
                'Estimated_Gap': estimated_gap,
                'IsOvertake': overtake_happened
            })
    
    relative_df = pd.DataFrame(relative_data)
    print(f"[OK] Creati {len(relative_df)} record relativi")
    
    return relative_df


def analyze_dataset_balance(df):
    """Analizza lo sbilanciamento delle classi."""
    overtakes = df['IsOvertake'].sum()
    total = len(df)
    perc = (overtakes / total) * 100
    
    print(f"\n[INFO] Analisi Dataset:")
    print(f"       Totale campioni: {total}")
    print(f"       Sorpassi: {overtakes} ({perc:.2f}%)")
    print(f"       Non sorpassi: {total - overtakes} ({100 - perc:.2f}%)")
    
    return perc


def main():
    print("=" * 60)
    print("RELATIVE FEATURE BUILDER - F1 Overtake Prediction")
    print("=" * 60)
    
    # 1. Carica dati
    df = load_raw_data()
    if df is None:
        return
    
    # 2. Costruisci feature relative
    relative_df = build_relative_features(df)
    
    if len(relative_df) == 0:
        print("[ERRORE] Nessun dato relativo creato.")
        return
    
    # 3. Analizza bilanciamento
    analyze_dataset_balance(relative_df)
    
    # 4. Salva
    output_path = '../../data/f1_monza_relative_features.csv'
    relative_df.to_csv(output_path, index=False)
    
    print(f"\n[OK] Dataset salvato: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
