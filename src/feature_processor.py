"""
================================================================================
FEATURE PROCESSOR - F1 Overtake Prediction
================================================================================
Questo script preprocessa le feature relative per il training del modello.

Operazioni eseguite:
1. Caricamento del dataset con feature relative
2. Selezione delle feature rilevanti per il modello
3. Split train/test stratificato (80/20)
4. Scaling con StandardScaler
5. Bilanciamento classi con SMOTE (Synthetic Minority Oversampling)

Tecniche ML applicate:
- Gestione missing values (fillna)
- Feature scaling (StandardScaler)
- Gestione dati sbilanciati (SMOTE da imbalanced-learn)
- Stratified split per mantenere proporzioni classi

Input: data/f1_monza_relative_features.csv
Output: 
- data/processed/X_train.npy, X_test.npy, y_train.npy, y_test.npy
- models/scaler.pkl, feature_names.pkl
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import pickle


def load_relative_features():
    """Carica il dataset con feature relative."""
    data_path = '../data/f1_monza_relative_features.csv'
    
    if not os.path.exists(data_path):
        print("[ERRORE] Dataset relativo non trovato! Esegui prima relative_feature_builder.py")
        return None
    
    df = pd.read_csv(data_path)
    print(f"[OK] Dataset caricato: {df.shape}")
    
    return df


def prepare_features(df):
    """Prepara le feature per il training."""
    print("\n[INFO] Preparazione features...")
    
    # Feature selezionate per il modello
    feature_columns = [
        'Attacker_Position',      # Posizione dell'attaccante
        'Delta_LapTime',          # Differenza tempo sul giro (negativo = piu veloce)
        'Delta_TyreLife',         # Differenza usura gomme
        'Compound_Advantage',     # Vantaggio tipo gomma
        'Attacker_LapTime',       # Tempo sul giro attaccante
        'Attacker_TyreLife',      # Usura gomme attaccante
    ]
    
    target_column = 'IsOvertake'
    
    # Verifica che tutte le colonne esistano
    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        print(f"[WARN] Colonne mancanti: {missing_cols}")
        return None, None
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Gestione valori mancanti
    X = X.fillna(0)
    
    print(f"       Feature utilizzate: {feature_columns}")
    print(f"       Dimensioni X: {X.shape}")
    
    return X, y


def split_and_scale(X, y):
    """Split dei dati e scaling."""
    print("\n[INFO] Split Train/Test...")
    
    # Stratified split per mantenere la proporzione delle classi
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback se troppi pochi esempi
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"       Training set: {X_train.shape[0]} campioni")
    print(f"       Test set: {X_test.shape[0]} campioni")
    print(f"       Sorpassi nel training: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
    
    # Scaling con StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def apply_smote(X_train, y_train):
    """Applica SMOTE per bilanciare le classi."""
    print("\n[INFO] Bilanciamento classi con SMOTE...")
    
    n_minority = y_train.sum()
    n_majority = len(y_train) - n_minority
    
    print(f"       Prima: {n_minority} positivi / {n_majority} negativi")
    
    try:
        # k_neighbors deve essere <= numero di campioni minoritari
        k_neighbors = min(5, n_minority - 1)
        if k_neighbors < 1:
            print("       [WARN] Troppi pochi campioni per SMOTE, skip...")
            return X_train, y_train
        
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        
        print(f"       Dopo: {y_res.sum()} positivi / {len(y_res) - y_res.sum()} negativi")
        print(f"       Dataset aumentato da {len(y_train)} a {len(y_res)} campioni")
        
        return X_res, y_res
    except Exception as e:
        print(f"       [WARN] SMOTE fallito: {e}")
        return X_train, y_train


def save_processed_data(X_train, X_test, y_train, y_test, scaler, feature_names):
    """Salva i dati processati e lo scaler."""
    print("\n[INFO] Salvataggio dati...")
    
    # Crea cartelle se necessario
    models_dir = '../models'
    processed_dir = '../data/processed'
    
    for d in [models_dir, processed_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # Salva arrays numpy
    np.save(f'{processed_dir}/X_train.npy', X_train)
    np.save(f'{processed_dir}/X_test.npy', X_test)
    np.save(f'{processed_dir}/y_train.npy', y_train)
    np.save(f'{processed_dir}/y_test.npy', y_test)
    
    # Salva scaler
    with open(f'{models_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Salva nomi delle feature (per reference)
    with open(f'{models_dir}/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print(f"       [OK] Dati salvati in {processed_dir}")
    print(f"       [OK] Scaler salvato in {models_dir}")


def main():
    print("=" * 60)
    print("FEATURE PROCESSOR - F1 Overtake Prediction")
    print("=" * 60)
    
    # 1. Carica dati
    df = load_relative_features()
    if df is None:
        return
    
    # 2. Prepara features
    X, y = prepare_features(df)
    if X is None:
        return
    
    # 3. Split e scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    
    # 4. SMOTE per bilanciamento
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    
    # 5. Salva
    feature_names = list(X.columns)
    save_processed_data(
        X_train_balanced, X_test, 
        y_train_balanced, y_test.values, 
        scaler, feature_names
    )
    
    print("\n" + "=" * 60)
    print("[OK] Preprocessing completato!")
    print("     Prossimo step: python model_trainer.py")
    print("=" * 60)


if __name__ == "__main__":
    main()