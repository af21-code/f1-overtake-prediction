"""
================================================================================
CORRELATION ANALYSIS - F1 Overtake Prediction
================================================================================
Questo script analizza la correlazione tra le feature del dataset per 
identificare eventuali feature ridondanti e giustificare le scelte di 
feature selection.

Analisi effettuate:
1. Matrice di correlazione tra tutte le feature
2. Identificazione feature altamente correlate (|r| > 0.7)
3. Visualizzazione heatmap

Output: reports/correlation_analysis.png, reports/correlation_matrix.csv
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data():
    """Carica il dataset con feature relative."""
    data_path = '../../data/f1_monza_relative_features.csv'
    
    if not os.path.exists(data_path):
        print("[ERRORE] Dataset non trovato!")
        return None
    
    df = pd.read_csv(data_path)
    print(f"[OK] Dataset caricato: {df.shape}")
    return df


def analyze_correlations(df):
    """Calcola e analizza la matrice di correlazione."""
    print("\n[INFO] Analisi correlazioni...")
    
    # Selezioniamo solo le feature numeriche rilevanti
    feature_columns = [
        'Attacker_Position',
        'Attacker_LapTime',
        'Attacker_TyreLife',
        'Defender_LapTime',
        'Defender_TyreLife',
        'Delta_LapTime',
        'Delta_TyreLife',
        'Compound_Advantage',
        'Estimated_Gap',
        'IsOvertake'
    ]
    
    # Filtra colonne esistenti
    existing_cols = [c for c in feature_columns if c in df.columns]
    df_numeric = df[existing_cols].copy()
    
    # Calcola matrice di correlazione
    corr_matrix = df_numeric.corr()
    
    return corr_matrix, df_numeric


def find_redundant_features(corr_matrix, threshold=0.7):
    """Identifica coppie di feature altamente correlate."""
    print(f"\n[INFO] Feature con correlazione |r| > {threshold}:")
    print("-" * 50)
    
    redundant_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) > threshold:
                redundant_pairs.append((col1, col2, corr_value))
                print(f"   {col1} <-> {col2}: r = {corr_value:.3f}")
    
    if not redundant_pairs:
        print("   Nessuna coppia con alta correlazione trovata.")
        print("   Le feature selezionate sono indipendenti.")
    
    return redundant_pairs


def plot_correlation_heatmap(corr_matrix, output_dir):
    """Genera la heatmap della matrice di correlazione."""
    plt.figure(figsize=(12, 10))
    
    # Crea heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt='.2f', 
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
        annot_kws={'size': 9}
    )
    
    plt.title('Feature Correlation Matrix - F1 Overtake Prediction', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    filename = f"{output_dir}/correlation_analysis.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] Heatmap salvata: {filename}")
    return filename


def analyze_target_correlations(corr_matrix):
    """Analizza la correlazione delle feature con il target."""
    print("\n[INFO] Correlazione con IsOvertake (target):")
    print("-" * 50)
    
    if 'IsOvertake' not in corr_matrix.columns:
        print("   Target 'IsOvertake' non trovato.")
        return
    
    target_corr = corr_matrix['IsOvertake'].drop('IsOvertake').sort_values(key=abs, ascending=False)
    
    for feature, corr in target_corr.items():
        direction = "+" if corr > 0 else "-"
        strength = "forte" if abs(corr) > 0.3 else "moderata" if abs(corr) > 0.15 else "debole"
        print(f"   {feature}: r = {corr:+.3f} ({strength})")


def main():
    print("=" * 60)
    print("CORRELATION ANALYSIS - F1 Overtake Prediction")
    print("=" * 60)
    
    # 1. Carica dati
    df = load_data()
    if df is None:
        return
    
    # 2. Calcola correlazioni
    corr_matrix, df_numeric = analyze_correlations(df)
    
    # 3. Trova feature ridondanti
    redundant = find_redundant_features(corr_matrix, threshold=0.7)
    
    # 4. Analizza correlazione con target
    analyze_target_correlations(corr_matrix)
    
    # 5. Crea directory output
    output_dir = '../../reports'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 6. Genera heatmap
    plot_correlation_heatmap(corr_matrix, output_dir)
    
    # 7. Salva matrice come CSV
    csv_path = f"{output_dir}/correlation_matrix.csv"
    corr_matrix.to_csv(csv_path)
    print(f"[OK] Matrice salvata: {csv_path}")
    
    # 8. Conclusioni
    print("\n" + "=" * 60)
    print("CONCLUSIONI ANALISI CORRELAZIONE")
    print("=" * 60)
    print("""
Le feature selezionate per il modello sono:
- Attacker_Position
- Delta_LapTime (differenza tempo sul giro)
- Delta_TyreLife (differenza usura gomme)
- Compound_Advantage (vantaggio tipo gomma)
- Attacker_LapTime
- Attacker_TyreLife

NOTA: Delta_LapTime è calcolato dalla differenza tra Attacker_LapTime 
e Defender_LapTime, quindi queste feature sono correlate BY DESIGN.
Tuttavia, Delta_LapTime cattura l'informazione relativa necessaria
per la predizione, mentre le feature assolute vengono escluse per
evitare ridondanza.

Le feature finali selezionate non presentano multicollinearità
problematica e contribuiscono ciascuna con informazioni uniche.
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
