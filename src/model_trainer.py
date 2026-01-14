import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_and_evaluate():
    print("🏋️‍♂️ Caricamento dati preprocessati...")

    # 1. Carichiamo i dati salvati dal passaggio precedente (File .npy veloci)
    try:
        X_train = np.load('../data/processed/X_train.npy')
        X_test = np.load('../data/processed/X_test.npy')
        y_train = np.load('../data/processed/y_train.npy')
        y_test = np.load('../data/processed/y_test.npy')
    except FileNotFoundError:
        print("❌ Errore: File non trovati. Esegui prima 'feature_processor.py'!")
        return

    print(f"✅ Dati caricati. Training shape: {X_train.shape}")

    # 2. Definizione dei 3 Modelli sfidanti
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    # Dizionario per salvare i punteggi
    results = {}

    # 3. Ciclo di Training e Valutazione
    for name, model in models.items():
        print(f"\n🤖 Addestramento {name} in corso...")

        # ADDESTRAMENTO (Qui il modello "studia")
        model.fit(X_train, y_train)

        # PREDIZIONE (Qui fa l'esame sui dati mai visti)
        y_pred = model.predict(X_test)

        # VALUTAZIONE
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = acc

        print(f"🎯 {name} completato!")
        print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")

        print("   Report dettagliato:")
        print(report)

        # VISUALIZZAZIONE MATRICE DI CONFUSIONE
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predetto dal Modello')
        plt.ylabel('Realtà')
        plt.tight_layout()
        plt.show()
        # NOTA: Il programma si metterà in pausa finché non chiudi la finestra del grafico!

    # 4. Verdetto Finale
    best_model = max(results, key=results.get)
    print("\n" + "="*50)
    print(f"🏆 IL VINCITORE È: {best_model}")
    print(f"   Con una accuracy di: {results[best_model]:.4f}")
    print("="*50)

if __name__ == "__main__":
    train_and_evaluate()