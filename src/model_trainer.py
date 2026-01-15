import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import pickle

def train_and_evaluate():
    print("🏋️‍♂️ Caricamento dati preprocessati...")

    try:
        X_train = np.load('../data/processed/X_train.npy')
        X_test = np.load('../data/processed/X_test.npy')
        y_train = np.load('../data/processed/y_train.npy')
        y_test = np.load('../data/processed/y_test.npy')
    except FileNotFoundError:
        print("❌ Errore: File non trovati. Esegui prima 'feature_processor.py'!")
        return

    print(f"✅ Dati caricati.")
    print(f"   Training shape: {X_train.shape}")
    print(f"   Test shape:     {X_test.shape}")

    # Calcoliamo il peso per bilanciare (utile per XGBoost)
    # scale_pos_weight = (numero negativi) / (numero positivi)
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

    # Definizione dei modelli
    # Aggiungiamo 'class_weight' per dire al modello: "I sorpassi sono importanti!"
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),

        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),

        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=ratio, use_label_encoder=False)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n🤖 ---------------- {name} ----------------")
        print("   Addestramento in corso...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        trained_models[name] = model

        print(f"   🎯 Accuracy: {acc:.2%}")

        # STAMPIAMO IL REPORT (Fondamentale!)
        print("\n   📊 Report Dettagliato:")
        print(classification_report(y_test, y_pred, target_names=['No Sorpasso', 'Sorpasso']))

        # Matrice di confusione sintetica
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"   ✅ Sorpassi presi (TP): {tp}")
        print(f"   ❌ Sorpassi persi (FN): {fn}")

    # Verdetto Finale
    best_model_name = max(results, key=results.get)
    best_acc = results[best_model_name]

    print("\n" + "="*50)
    print(f"🏆 IL VINCITORE È: {best_model_name}")
    print(f"   Con una accuracy di: {best_acc:.2%}")
    print("="*50)

    # --- SALVATAGGIO DEL MODELLO VINCITORE ---
    print(f"💾 Salvataggio di {best_model_name} in corso...")

    if not os.path.exists('../models'):
        os.makedirs('../models')

    final_model = trained_models[best_model_name]

    with open('../models/best_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)

    print(f"✅ Modello salvato in: ../models/best_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()