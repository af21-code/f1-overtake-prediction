import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import os
import pickle

def smart_rename_columns(df):
    """
    Rinomina colonne standard, ma NON blocca tutto se manca il nome della gara.
    """
    mapping = {
        'Year': ['Season'],
        'LapNumber': ['Lap', 'Laps'],
        'Driver': ['DriverNumber', 'Pilot'],
        'Compound': ['Tyre', 'Tires']
    }

    print("\n🔍 Controllo colonne...")
    for target_col, possibilities in mapping.items():
        if target_col not in df.columns:
            for candidate in possibilities:
                if candidate in df.columns:
                    print(f"   ⚠️ Rinomino '{candidate}' in '{target_col}'")
                    df.rename(columns={candidate: target_col}, inplace=True)
                    break

    return df

def prepare_data_for_training():
    print("⚙️ Inizio Preprocessing avanzato...")

    # 1. Carichiamo il dataset
    try:
        df = pd.read_csv('../data/f1_ground_effect_dataset.csv')
        print(f"✅ Dataset caricato: {df.shape}")
        print(f"   Colonne trovate: {list(df.columns)}")
    except FileNotFoundError:
        print("❌ Errore: File CSV non trovato.")
        return

    # 2. Rinomina colonne (se necessario)
    df = smart_rename_columns(df)

    # 3. ORDINAMENTO
    # Non avendo il nome della gara, ordiniamo per Anno -> Pilota -> Giro
    print("🔄 Ordinamento dati...")
    sort_cols = [c for c in ['Year', 'Driver', 'LapNumber'] if c in df.columns]
    df = df.sort_values(by=sort_cols)

    # 4. FEATURE ENGINEERING
    print("🛠️ Creazione features...")

    # A. Convertiamo LapTime
    # Alcuni dataset hanno LapTime già in secondi (float), altri stringa. Controlliamo.
    if df['LapTime'].dtype == object:
        # Se c'è "0 days", lo togliamo per evitare errori di parsing
        df['LapTime'] = df['LapTime'].astype(str).str.replace('0 days ', '', regex=False)
        df['LapTime_Sec'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
    else:
        # Se è già float, usalo così com'è
        df['LapTime_Sec'] = df['LapTime']

    # B. Calcolo NextPosition e IsOvertake
    # Raggruppiamo per Anno e Pilota (visto che manca il GP)
    group_cols = [c for c in ['Year', 'Driver'] if c in df.columns]

    # Prendiamo la posizione e il numero di giro della riga successiva
    df['NextPosition'] = df.groupby(group_cols)['Position'].shift(-1)
    df['NextLapNumber'] = df.groupby(group_cols)['LapNumber'].shift(-1)

    # --- PROTEZIONE DATA LEAKAGE ---
    # Se il prossimo giro NON è (Giro Attuale + 1), significa che è iniziata un'altra gara!
    # In quel caso, non possiamo prevedere nulla.
    df = df[df['NextLapNumber'] == df['LapNumber'] + 1]

    # Calcolo Target
    df['IsOvertake'] = (df['Position'] > df['NextPosition']).astype(int)

    # 5. PULIZIA
    # Rimuoviamo colonne non numeriche o ausiliarie
    cols_to_drop = ['Driver', 'Team', 'GP_Name', 'LapTime', 'NextPosition', 'NextLapNumber', 'Year']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 6. ENCODING GOMME
    le = LabelEncoder()
    if 'Compound' in df.columns:
        df['Compound'] = df['Compound'].astype(str)
        df['Compound'] = le.fit_transform(df['Compound'])

        if not os.path.exists('../models'):
            os.makedirs('../models')
        with open('../models/le_compound.pkl', 'wb') as f:
            pickle.dump(le, f)
        print("✅ Encoder gomme salvato.")

    # 7. MISSING VALUES & FINAL CHECK
    df = df.fillna(0)

    # Check se abbiamo ancora dati
    if len(df) == 0:
        print("❌ ERRORE: Il dataset è vuoto dopo il preprocessing. Controlla la logica dei giri.")
        return

    # 8. SPLIT
    X = df.drop(columns=['IsOvertake'])
    y = df['IsOvertake']

    # Stratify è importante, ma se abbiamo pochissimi sorpassi potrebbe fallire.
    # Aggiungiamo un try/except per sicurezza
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback se ci sono troppi pochi esempi di una classe
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    print(f"\n📊 Dati Training: {X_train.shape} | Sorpassi totali: {y_train.sum()}")

    # 9. SMOTE (Gestione sbilanciamento)
    try:
        # k_neighbors=1 serve se hai pochissimi dati
        smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"✅ SMOTE applicato! Training set aumentato a: {X_train_res.shape}")
    except Exception as e:
        print(f"⚠️ SMOTE saltato ({e}). Uso dati originali.")
        X_train_res, y_train_res = X_train, y_train

    # 10. SCALING
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    with open('../models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 11. SALVATAGGIO
    if not os.path.exists('../data/processed'):
        os.makedirs('../data/processed')

    np.save('../data/processed/X_train.npy', X_train_res)
    np.save('../data/processed/X_test.npy', X_test)
    np.save('../data/processed/y_train.npy', y_train_res)
    np.save('../data/processed/y_test.npy', y_test)

    print("\n💾 Tutto salvato in ../data/processed/")

if __name__ == "__main__":
    prepare_data_for_training()