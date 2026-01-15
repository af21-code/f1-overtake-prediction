import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import os

def prepare_data_for_training():
    print("⚙️ Inizio Preprocessing avanzato...")

    # 1. Carichiamo il dataset grezzo
    try:
        df = pd.read_csv('../data/f1_2023_processed.csv')
    except FileNotFoundError:
        print("❌ Errore: File CSV non trovato. Esegui prima 'data_loader.py'!")
        return

    # --- NUOVO: CALCOLO DEL TARGET E PULIZIA DATI ---

    # A. Convertiamo LapTime da stringa a secondi (necessario per l'IA)
    # Il formato è solitamente "0 days 00:01:25.123", lo trasformiamo in float
    df['LapTime_Sec'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()

    # B. Creiamo la colonna NextPosition e IsOvertake (Feature Engineering)
    # Raggruppiamo per pilota e spostiamo la posizione indietro di 1 per vedere il futuro
    df['NextPosition'] = df.groupby('Driver')['Position'].shift(-1)

    # Se la posizione attuale è > della prossima (es. ero 5°, ora sono 4°), è un sorpasso (1)
    df['IsOvertake'] = (df['Position'] > df['NextPosition']).astype(int)

    # Rimuoviamo l'ultimo giro di ogni pilota (non ha un "prossimo giro")
    df = df.dropna(subset=['NextPosition'])

    # -----------------------------------------------

    # 2. Pulizia Extra (Rimuoviamo colonne inutili per il modello)
    # Usiamo errors='ignore' così se 'Team' non c'è, non crasha!
    cols_to_drop = ['Driver', 'Team', 'GP_Name', 'LapTime', 'NextPosition', 'LapNumber']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 3. ENCODING: Trasformiamo le parole in numeri
    # Esempio: SOFT -> 0, MEDIUM -> 1, HARD -> 2
    le = LabelEncoder()
    if 'Compound' in df.columns:
        df['Compound'] = le.fit_transform(df['Compound'])
        print("✅ Gomme convertite in numeri.")

    # 4. GESTIONE MISSING VALUES
    df = df.fillna(0)

    # 5. SEPARAZIONE FEATURES (X) e TARGET (y)
    X = df.drop(columns=['IsOvertake'])  # Tutto tranne il risultato
    y = df['IsOvertake']                 # Solo il risultato (1 o 0)

    # 6. SPLIT TRAIN/TEST
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\n📊 Dati Training: {X_train.shape}")
    print(f"   Sorpassi originali: {y_train.sum()}")

    # 7. RISOLUZIONE SBILANCIAMENTO (SMOTE)
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"✅ SMOTE applicato! Sorpassi bilanciati: {y_train_res.sum()}")
    except ValueError:
        print("⚠️ Attenzione: Troppi pochi dati per SMOTE. Uso i dati originali.")
        X_train_res, y_train_res = X_train, y_train

    # 8. SCALING (Normalizzazione)
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    # 9. SALVATAGGIO FILE PRONTI
    if not os.path.exists('../data/processed'):
        os.makedirs('../data/processed')

    np.save('../data/processed/X_train.npy', X_train_res)
    np.save('../data/processed/X_test.npy', X_test)
    np.save('../data/processed/y_train.npy', y_train_res)
    np.save('../data/processed/y_test.npy', y_test)

    print("\n💾 File di training salvati in ../data/processed/")

if __name__ == "__main__":
    prepare_data_for_training()