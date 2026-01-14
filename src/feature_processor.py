import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import os

def prepare_data_for_training():
    print("⚙️ Inizio Preprocessing avanzato...")

    # 1. Carichiamo il dataset grezzo
    df = pd.read_csv('../data/f1_2023_processed.csv')

    # 2. Pulizia Extra (Rimuoviamo colonne inutili per il modello)
    # MODIFICA FONDAMENTALE: Rimuoviamo 'NextPosition' per evitare Data Leakage!
    # Togliamo anche Driver/Team (bias) e LapTime (usiamo i secondi).
    df = df.drop(columns=['Driver', 'Team', 'GP_Name', 'LapTime', 'NextPosition'])

    # 3. ENCODING: Trasformiamo le parole in numeri
    # Esempio: SOFT -> 0, MEDIUM -> 1, HARD -> 2
    le = LabelEncoder()
    if 'Compound' in df.columns:
        df['Compound'] = le.fit_transform(df['Compound'])
        print("✅ Gomme convertite in numeri.")

    # 4. GESTIONE MISSING VALUES
    # Se manca qualche numero, lo riempiamo con zero
    df = df.fillna(0)

    # 5. SEPARAZIONE FEATURES (X) e TARGET (y)
    X = df.drop(columns=['IsOvertake'])  # Tutto tranne il risultato
    y = df['IsOvertake']                 # Solo il risultato (1 o 0)

    # 6. SPLIT TRAIN/TEST (Temporale)
    # 80% per addestramento, 20% per test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\n📊 Dati originali Training: {X_train.shape}")
    print(f"   Sorpassi nel training (prima): {y_train.sum()}")

    # 7. RISOLUZIONE SBILANCIAMENTO (SMOTE)
    # Creiamo dati sintetici per bilanciare le classi
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"✅ SMOTE applicato!")
    print(f"   Sorpassi nel training (dopo): {y_train_res.sum()}")
    print(f"   Totale righe training: {X_train_res.shape[0]}")

    # 8. SCALING (Normalizzazione)
    # Portiamo tutti i numeri sulla stessa scala
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test) # Usiamo lo stesso scaler del train

    # 9. SALVATAGGIO FILE PRONTI
    if not os.path.exists('../data/processed'):
        os.makedirs('../data/processed')

    # Salviamo in formato numpy (veloce)
    np.save('../data/processed/X_train.npy', X_train_res)
    np.save('../data/processed/X_test.npy', X_test)
    np.save('../data/processed/y_train.npy', y_train_res)
    np.save('../data/processed/y_test.npy', y_test)

    print("\n💾 File di training salvati in ../data/processed/")

if __name__ == "__main__":
    prepare_data_for_training()