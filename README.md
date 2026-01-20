# F1 Overtake Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Sistema di Machine Learning per predire la probabilità di sorpasso in Formula 1**

*Progetto per il corso di Machine Learning - A.A. 2025/26*

[Installazione](#installazione) •
[Come Usare](#come-usare) •
[Struttura Progetto](#struttura-progetto) •
[Metodologia](#metodologia) •
[Risultati](#risultati)

</div>

---

## Panoramica

Questo progetto implementa un sistema di **Machine Learning** per predire la probabilità che un pilota effettui un sorpasso durante una gara di Formula 1. L'applicazione simula un **muretto box** durante il Gran Premio di Monza 2025, permettendo agli ingegneri di valutare scenari di sorpasso in tempo reale.

### Caratteristiche Principali

- **Dati Reali**: Utilizzo di dati telemetrici ufficiali F1 tramite FastF1 (Monza 2022-2024)
- **Feature Engineering**: Calcolo di metriche relative tra piloti (delta tempo, usura gomme, vantaggio compound)
- **Confronto Modelli**: Valutazione di Logistic Regression, Random Forest e XGBoost
- **Web Application**: Interfaccia professionale per simulare scenari di sorpasso
- **Gestione Sbilanciamento**: Applicazione di SMOTE per bilanciare le classi

---

## Installazione

### Prerequisiti

- Python 3.9 o superiore
- pip (package manager)

### Setup

```bash
# Clona il repository
git clone https://github.com/username/f1-overtake-prediction.git
cd f1-overtake-prediction

# Crea ambiente virtuale
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Installa dipendenze
pip install -r requirements.txt
```

---

## Come Usare

### 1. Esegui la Pipeline Completa

```bash
cd src

# Step 1: Scarica dati da FastF1 (richiede connessione internet)
python data_loader.py

# Step 2: Costruisci feature relative
python relative_feature_builder.py

# Step 3: Preprocessa i dati
python feature_processor.py

# Step 4: Addestra i modelli
python model_trainer.py
```

### 2. Avvia la Web Application

```bash
cd src
streamlit run app.py
```

L'applicazione sarà disponibile su `http://localhost:8501`

### Utilizzo della Webapp

1. **Inserisci i dati del tuo pilota**: posizione, tipo gomma, usura, tempo sul giro
2. **Inserisci i dati dell'avversario**: tipo gomma, usura, tempo sul giro
3. **Imposta il gap** tra i due piloti
4. **Clicca "CALCULATE PROBABILITY"** per ottenere la predizione
5. Usa **SWAP** per passare dalla modalità ATTACK a DEFENSE

---

## Struttura Progetto

```
f1-overtake-prediction/
│
├── data/
│   ├── f1_ground_effect_dataset.csv      # Dati grezzi Monza 2022-2024
│   ├── f1_monza_relative_features.csv    # Feature relative processate
│   └── processed/                         # Dati pronti per training
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
│
├── models/
│   ├── best_model.pkl                    # Modello addestrato
│   ├── scaler.pkl                        # StandardScaler
│   ├── feature_names.pkl                 # Nomi delle feature
│   └── model_info.json                   # Metadati modello
│
├── reports/
│   ├── training_report.json              # Metriche complete
│   ├── confusion_matrix_*.png            # Matrici di confusione
│   ├── roc_curves.png                    # Curve ROC comparative
│   └── feature_importance_*.png          # Importanza feature
│
├── src/
│   ├── data_loader.py                    # Download dati FastF1
│   ├── relative_feature_builder.py       # Costruzione feature relative
│   ├── feature_processor.py              # Preprocessing e SMOTE
│   ├── model_trainer.py                  # Training e valutazione
│   └── app.py                            # Web application Streamlit
│
├── notebooks/                            # Analisi esplorativa (EDA)
├── cache/                                # Cache FastF1 (gitignored)
├── requirements.txt                      # Dipendenze Python
├── LICENSE                               # Licenza MIT
└── README.md                             # Documentazione
```

---

## Metodologia

### 1. Data Collection

I dati vengono estratti utilizzando **FastF1**, una libreria Python per accedere ai dati telemetrici ufficiali della Formula 1. Vengono considerati i Gran Premi di Monza delle stagioni 2022, 2023 e 2024.

**Dati estratti per ogni giro:**
- `Driver`: codice pilota
- `LapNumber`: numero del giro
- `LapTime`: tempo sul giro
- `Position`: posizione in classifica
- `TyreLife`: giri con il set di gomme attuale
- `Compound`: tipo di mescola (SOFT/MEDIUM/HARD)

### 2. Feature Engineering

Per predire un sorpasso, è fondamentale costruire **feature relative** che catturino la dinamica del duello tra attaccante e difensore:

| Feature | Descrizione |
|---------|-------------|
| `Delta_LapTime` | Differenza tempo sul giro (negativo = attaccante più veloce) |
| `Delta_TyreLife` | Differenza usura gomme (negativo = gomme più fresche) |
| `Compound_Advantage` | Vantaggio tipo gomma (SOFT=3, MEDIUM=2, HARD=1) |
| `Attacker_Position` | Posizione dell'attaccante |
| `Attacker_LapTime` | Tempo sul giro dell'attaccante |
| `Attacker_TyreLife` | Usura gomme dell'attaccante |

**Target Variable:** `IsOvertake` (1 se l'attaccante ha guadagnato posizione nel giro successivo)

### 3. Preprocessing

- **Outlier Detection**: Rimozione giri anomali (pit stop, safety car) usando deviazione standard
- **Missing Values**: Gestione con `fillna(0)`
- **Scaling**: StandardScaler per normalizzare le feature
- **SMOTE**: Synthetic Minority Oversampling per bilanciare le classi (sorpassi sono eventi rari)

### 4. Model Selection

Tre modelli vengono confrontati:

| Modello | Configurazione |
|---------|----------------|
| **Logistic Regression** | `class_weight='balanced'`, `max_iter=1000` |
| **Random Forest** | `n_estimators=100`, `class_weight='balanced'` |
| **XGBoost** | `eval_metric='logloss'` |

### 5. Evaluation Metrics

- **Accuracy**: Percentuale predizioni corrette
- **Precision**: TP / (TP + FP) - affidabilità delle predizioni positive
- **Recall**: TP / (TP + FN) - capacità di trovare tutti i positivi
- **F1-Score**: Media armonica di precision e recall
- **ROC-AUC**: Area sotto la curva ROC

Il modello con il **miglior F1-Score** viene selezionato.

---

## Risultati

### Performance dei Modelli

| Modello | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.712 | 0.321 | **0.630** | **0.425** | 0.750 |
| Random Forest | 0.800 | 0.405 | 0.395 | 0.400 | 0.778 |
| XGBoost | 0.814 | 0.441 | 0.370 | 0.403 | 0.757 |

**Modello Selezionato:** Logistic Regression (miglior compromesso tra Precision e Recall)

### Interpretazione

La Logistic Regression è stata selezionata nonostante l'accuracy inferiore perché:
- **Alto Recall (0.630)**: Cattura la maggior parte dei sorpassi reali
- **Interpretabilità**: I coefficienti possono essere analizzati per capire il contributo di ogni feature
- **Robustezza**: Meno prone a overfitting rispetto a modelli più complessi

---

## Tech Stack

| Categoria | Tecnologie |
|-----------|------------|
| **Linguaggio** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **F1 Data** | FastF1 |
| **Machine Learning** | Scikit-Learn, Imbalanced-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Web App** | Streamlit |

---

## Autori

**Angelo Fusco** e **Mattia Fanzini**

Progetto sviluppato per il corso di *Machine Learning*  
Professori: G. Polese & L. Caruccio  
Anno Accademico 2025/26

---

## Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi il file [LICENSE](LICENSE) per i dettagli.
