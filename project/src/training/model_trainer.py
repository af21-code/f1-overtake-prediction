"""
================================================================================
MODEL TRAINER - F1 Overtake Prediction
================================================================================
Questo script addestra e valuta diversi modelli di classificazione per predire
la probabilita di sorpasso in Formula 1.

Modelli confrontati:
1. Logistic Regression (con class_weight='balanced')
2. Random Forest (con class_weight='balanced')
3. XGBoost (con scale_pos_weight per bilanciamento)

Metriche di valutazione:
- Accuracy: percentuale predizioni corrette
- Precision: TP / (TP + FP) - affidabilita delle predizioni positive
- Recall: TP / (TP + FN) - capacita di trovare tutti i positivi
- F1-Score: media armonica di precision e recall
- ROC-AUC: area sotto la curva ROC

Il modello con la miglior Accuracy viene selezionato e salvato.

Input: data/processed/X_train.npy, X_test.npy, y_train.npy, y_test.npy
Output:
- models/best_model.pkl
- models/model_info.json
- reports/training_report.json
- reports/confusion_matrix_*.png
- reports/roc_curves.png
- reports/feature_importance_*.png
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non interattivo per server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import os
import pickle
import json


def load_processed_data():
    """Carica i dati preprocessati."""
    processed_dir = '../../data/processed'
    
    try:
        X_train = np.load(f'{processed_dir}/X_train.npy')
        X_test = np.load(f'{processed_dir}/X_test.npy')
        y_train = np.load(f'{processed_dir}/y_train.npy')
        y_test = np.load(f'{processed_dir}/y_test.npy')
        
        print(f"[OK] Dati caricati:")
        print(f"     Training: {X_train.shape}")
        print(f"     Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"[ERRORE] File non trovato: {e}")
        print("         Esegui prima feature_processor.py!")
        return None, None, None, None


def get_models():
    """
    Configurazione dei modelli per il confronto.
    
    Tutti i modelli usano tecniche per gestire lo sbilanciamento delle classi:
    - class_weight='balanced': peso inverso alla frequenza della classe
    - max_iter=1000: convergenza garantita per dataset di questa dimensione
    - n_estimators=100: trade-off ottimale tra performance e tempo di training
    """
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42, 
            class_weight='balanced', 
            max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced',
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    }
    return models


def evaluate_model(model, X_test, y_test, model_name):
    """Valuta un modello e restituisce le metriche."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n[MODEL] {model_name}")
    print("-" * 40)
    print(f"   Accuracy:  {metrics['accuracy']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall:    {metrics['recall']:.3f}")
    print(f"   F1-Score:  {metrics['f1']:.3f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(f"\n   Confusion Matrix:")
    print(f"   TP: {tp} | FP: {fp}")
    print(f"   FN: {fn} | TN: {tn}")
    
    return metrics, cm, y_pred, y_prob


def plot_confusion_matrix(cm, model_name, output_dir):
    """Genera e salva la confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Sorpasso', 'Sorpasso'],
                yticklabels=['No Sorpasso', 'Sorpasso'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    filename = f"{output_dir}/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    
    return filename


def plot_roc_curves(all_results, y_test, output_dir):
    """Genera le curve ROC per tutti i modelli."""
    plt.figure(figsize=(10, 8))
    
    for model_name, results in all_results.items():
        fpr, tpr, _ = roc_curve(y_test, results['y_prob'])
        auc = results['metrics']['roc_auc']
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{output_dir}/roc_curves.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    
    return filename


def plot_feature_importance(model, feature_names, model_name, output_dir):
    """Genera il grafico dell'importanza delle feature (per modelli ad albero)."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        
        # Ordina per importanza
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance[indices], color='steelblue')
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title(f'Feature Importance - {model_name}')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        filename = f"{output_dir}/feature_importance_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        
        return filename
    return None


def train_and_evaluate():
    """Pipeline principale di training e valutazione."""
    print("=" * 60)
    print("MODEL TRAINER - F1 Overtake Prediction")
    print("=" * 60)
    
    # 1. Carica dati
    X_train, X_test, y_train, y_test = load_processed_data()
    if X_train is None:
        return
    
    # Carica nomi delle feature
    try:
        with open('../../models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
    except:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # 2. Crea directory per i grafici
    output_dir = '../../reports'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 3. Ottieni modelli
    models = get_models()
    
    # 4. Training e valutazione
    all_results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n[TRAIN] Training {name}...")
        model.fit(X_train, y_train)
        
        metrics, cm, y_pred, y_prob = evaluate_model(model, X_test, y_test, name)
        
        # Salva confusion matrix
        plot_confusion_matrix(cm, name, output_dir)
        
        # Feature importance (se disponibile)
        plot_feature_importance(model, feature_names, name, output_dir)
        
        all_results[name] = {
            'metrics': metrics,
            'cm': cm.tolist(),
            'y_prob': y_prob
        }
        trained_models[name] = model
    
    # 5. ROC curves comparative
    plot_roc_curves(all_results, y_test, output_dir)
    
    # 6. Trova il modello migliore (per Accuracy)
    best_model_name = max(all_results, key=lambda x: all_results[x]['metrics']['accuracy'])
    best_metrics = all_results[best_model_name]['metrics']
    
    print("\n" + "=" * 60)
    print(f"[BEST] MIGLIOR MODELLO: {best_model_name}")
    print(f"       F1-Score: {best_metrics['f1']:.3f}")
    print(f"       ROC-AUC:  {best_metrics['roc_auc']:.3f}")
    print("=" * 60)
    
    # 7. Salva il modello vincitore
    best_model = trained_models[best_model_name]
    
    models_dir = '../../models'
    with open(f'{models_dir}/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Salva anche il nome del modello per reference
    model_info = {
        'name': best_model_name,
        'metrics': best_metrics,
        'feature_names': feature_names
    }
    with open(f'{models_dir}/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # 8. Salva tutte le metriche per il report
    report_data = {
        'models': {name: {'metrics': results['metrics'], 'confusion_matrix': results['cm']} 
                   for name, results in all_results.items()},
        'best_model': best_model_name,
        'feature_names': feature_names
    }
    with open(f'{output_dir}/training_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n[OK] Modello salvato: {models_dir}/best_model.pkl")
    print(f"[OK] Report salvato: {output_dir}/training_report.json")
    print(f"[OK] Grafici salvati in: {output_dir}/")


if __name__ == "__main__":
    train_and_evaluate()