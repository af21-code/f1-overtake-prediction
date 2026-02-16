# 📘 Guida Esplicativa al Progetto - F1 Overtake Prediction

> **Nota**: Questo file non viene committato. Serve come guida per capire e spiegare il progetto durante la presentazione.

---

## 🎯 Obiettivo del Progetto

Creare un sistema di **Machine Learning** che predice la probabilità di sorpasso in Formula 1, simulando un muretto box durante una gara.

**Perché questo progetto?**
- I sorpassi in F1 sono eventi **rari ma cruciali** per le strategie di gara
- I team usano dati telemetrici per prendere decisioni in tempo reale
- È un problema di **classificazione binaria sbilanciata** (pochi sorpassi vs tanti non-sorpassi)

---

## 📁 Struttura e File

### `pipeline/data_loader.py`
**Cosa fa:** Scarica i dati telemetrici ufficiali F1 dal GP di Monza (2022-2024).

**Perché FastF1?**
- È l'unica libreria Python che accede ai dati ufficiali F1
- Fornisce dati giro per giro: tempo, posizione, gomme
- Dati reali = risultati più significativi rispetto a dati sintetici

**Perché solo Monza?**
- Circuito con molte opportunità di sorpasso (rettilineo principale, Curva Grande)
- Omogeneità dei dati (stesso circuito, condizioni simili)
- 3 anni = ~1500 giri utili per il training

---

### `pipeline/relative_feature_builder.py`
**Cosa fa:** Trasforma i dati grezzi in **feature relative** tra coppie di piloti.

**Perché feature relative?**
- Un sorpasso dipende dalla **differenza** tra due piloti, non dai valori assoluti
- Es: non importa se il pilota fa 1:24, importa se è 0.5s più veloce di chi ha davanti

**Feature create:**
| Feature | Significato | Perché è importante |
|---------|-------------|---------------------|
| `Delta_LapTime` | Differenza tempo sul giro | Negativo = attaccante più veloce |
| `Delta_TyreLife` | Differenza usura gomme | Gomme fresche = più grip |
| `Compound_Advantage` | SOFT=3, MEDIUM=2, HARD=1 | Mescola più morbida = più veloce |
| `Estimated_Gap` | Gap stimato | Influenza la probabilità di DRS |

**Perché il filtro outlier (2σ)?**
- Rimuove giri anomali: pit stop, safety car, bandiere gialle
- Questi eventi non rappresentano la normale dinamica di sorpasso

---

### `pipeline/feature_processor.py`
**Cosa fa:** Prepara i dati per il training.

**Scaling con StandardScaler - Perché?**
- Le feature hanno scale diverse (posizione: 1-20, tempo: 80-100s)
- StandardScaler normalizza a media=0, std=1
- Migliora convergenza dei modelli e performance di XGBoost

**SMOTE - Perché?**
- I sorpassi sono eventi rari (~8% dei campioni)
- Modelli tendono a predire sempre "no sorpasso" senza bilanciamento
- SMOTE crea campioni sintetici della classe minoritaria

**k_neighbors adattivo - Perché?**
- SMOTE standard usa k=5, ma fallisce se ci sono meno di 5 campioni positivi
- Adattiamo automaticamente k al numero di campioni disponibili

---

### `training/model_trainer.py`
**Cosa fa:** Addestra e confronta 3 modelli, seleziona il migliore.

**Perché questi 3 modelli?**

| Modello | Pro | Contro |
|---------|-----|--------|
| **Logistic Regression** | Interpretabile, veloce, robusto | Assume linearità |
| **Random Forest** | Cattura non-linearità, feature importance | Può overfittare |
| **XGBoost** | State-of-the-art, molto accurato | Black box, più lento |

**Perché Accuracy come metrica principale?**
- Con il bilanciamento delle classi tramite SMOTE e class_weight, l'accuracy diventa una metrica significativa
- XGBoost raggiunge la migliore accuracy (81.4%) grazie al gradient boosting

**Perché `class_weight='balanced'`?**
- Penalizza maggiormente gli errori sulla classe minoritaria
- Alternativa a SMOTE, li usiamo insieme per massimizzare l'effetto

---

### `app/app.py`
**Cosa fa:** Web application che simula il muretto box.

**Perché Streamlit?**
- Framework Python per creare webapp senza JavaScript
- Perfetto per demo di progetti ML
- Interfaccia reattiva con poche righe di codice

**Modalità ATTACK vs DEFENSE:**
- ATTACK: probabilità che il TUO pilota sorpassi
- DEFENSE: probabilità che il tuo pilota VENGA sorpassato
- Stessa logica, ma attaccante/difensore invertiti

---

### `analysis/correlation_analysis.py`
**Cosa fa:** Analizza le correlazioni tra feature.

**Perché questa analisi?**
- Identificare feature ridondanti (alta correlazione tra loro)
- Verificare correlazione con il target (IsOvertake)
- Giustificare la scelta delle feature

---

## 🔬 Scelte Tecniche Chiave

### 1. Perché XGBoost vince?

XGBoost ha la migliore accuracy (81.4%) e precision (0.44) tra i tre modelli:
- **Accuracy superiore (81.4% vs 71.2% LR)**: migliore capacità predittiva complessiva
- **Precision più alta (0.44 vs 0.32 LR)**: meno falsi positivi, predizioni più affidabili
- **Gradient Boosting**: cattura relazioni non lineari tra le feature

**Nel contesto F1:** Le predizioni di XGBoost sono più affidabili, riducendo i falsi allarmi al muretto box e fornendo suggerimenti più precisi.

### 2. Perché solo 6 feature?

- **Parsimonia**: meno feature = modello più generalizzabile
- **Interpretabilità**: ogni feature ha un significato chiaro
- **Evitare multicollinearità**: non includere sia Delta_LapTime che Attacker_LapTime + Defender_LapTime

### 3. Perché train/test 80/20?

- Standard nel ML
- Stratified split mantiene la proporzione delle classi
- Con ~1500 campioni, 20% = ~300 test samples, statisticamente significativo

---

## 📊 Interpretazione dei Risultati

### Confusion Matrix del modello finale (XGBoost):

```
                    Pred: No    Pred: Sì
Actual: No           360 (TN)    38 (FP)
Actual: Sì           51 (FN)     30 (TP)
```

**Lettura:**
- **30 True Positives**: sorpassi correttamente previsti ✅
- **51 False Negatives**: sorpassi mancati ❌
- **38 False Positives**: pochi falsi allarmi ⚠️ (alta precision)
- **360 True Negatives**: correttamente previsto "no sorpasso" ✅

---

## 🚀 Flusso di Esecuzione

```
1. data_loader.py
   └─→ Scarica dati FastF1 → f1_ground_effect_dataset.csv

2. relative_feature_builder.py
   └─→ Crea coppie pilota-avversario → f1_monza_relative_features.csv

3. feature_processor.py
   └─→ Scale + SMOTE → X_train.npy, X_test.npy, scaler.pkl

4. model_trainer.py
   └─→ Train 3 modelli → best_model.pkl, training_report.json

5. app.py
   └─→ Carica modello → Interfaccia utente interattiva
```

---

## 💡 Possibili Domande in Sede d'Esame

### 📌 Domande sul Dataset e Preprocessing

**Q: Perché avete scelto solo il circuito di Monza?**
> Per garantire **omogeneità dei dati**. Ogni circuito ha caratteristiche diverse (rettilinei, curve, possibilità di sorpasso). Mischiare circuiti introdurrebbe rumore e variabilità non legata alle feature del modello.

**Q: Perché usate solo 3 anni di dati?**
> Perché dal 2022 è iniziata l'era "Ground Effect" con nuove regole aerodinamiche. Usare dati precedenti includerebbe macchine con comportamenti diversi, rendendo il modello meno accurato.

**Q: Come gestite i valori mancanti?**
> Usiamo `fillna(0)` per i missing values. Questo è appropriato perché i missing tendono a essere giri incompleti (interruzioni) dove le feature non sono significative.

**Q: Perché rimuovete i giri oltre 2 deviazioni standard?**
> Per eliminare **outlier** come pit stop, safety car, partenze da fermo. Questi giri hanno tempi anomali che non rappresentano la normale dinamica di sorpasso.

**Q: Perché usate StandardScaler e non MinMaxScaler?**
> StandardScaler è preferibile per i nostri modelli (incluso **XGBoost**) perché normalizza assumendo distribuzione normale. MinMaxScaler è sensibile agli outlier e forza i valori in [0,1], perdendo informazione sulla distribuzione.

---

### 📌 Domande sul Feature Engineering

**Q: Perché usate feature relative invece di assolute?**
> Un sorpasso dipende dalla **differenza** tra due piloti. Non importa se un pilota fa 1:24, importa quanto è più veloce di chi ha davanti. Le feature relative catturano questa dinamica.

**Q: Cosa rappresenta Compound_Advantage?**
> È la differenza tra il valore numerico delle mescole (SOFT=3, MEDIUM=2, HARD=1). Un valore positivo indica che l'attaccante ha gomme più performanti.

**Q: Perché non avete usato altre feature come velocità massima o settori?**
> Per **parsimonia**. Più feature non significa modello migliore. Le 6 feature selezionate catturano le informazioni chiave senza rischio di overfitting e multicollinearità.

**Q: Come definite se un sorpasso è avvenuto?**
> Confrontiamo la posizione del pilota nel giro N+1 con quella nel giro N. Se l'attaccante ha guadagnato posizione, `IsOvertake = 1`.

---

### 📌 Domande sullo Sbilanciamento delle Classi

**Q: Quanto è sbilanciato il dataset?**
> Circa **8% sorpassi vs 92% non-sorpassi**. È fortemente sbilanciato perché i sorpassi sono eventi rari in F1.

**Q: Perché usate SMOTE?**
> SMOTE (Synthetic Minority Oversampling) genera campioni sintetici della classe minoritaria interpolando tra campioni esistenti. Questo bilancia il dataset senza perdere informazioni.

**Q: Perché usate anche class_weight='balanced'?**
> È una tecnica complementare a SMOTE. Penalizza maggiormente gli errori sulla classe minoritaria durante il training. Le usiamo entrambe per massimizzare l'effetto.

**Q: Cosa succede se non bilanciate le classi?**
> Il modello impara a predire sempre "no sorpasso" perché questa classe domina. Ottiene 92% accuracy ma 0% recall sui sorpassi, rendendolo inutile.

**Q: Perché k_neighbors in SMOTE è adattivo?**
> SMOTE standard usa k=5 vicini, ma fallisce se esistono meno di 5 campioni della classe minoritaria. Noi adattiamo k automaticamente a `min(5, n_minority - 1)`.

---

### 📌 Domande sui Modelli

**Q: Perché avete scelto questi 3 modelli specifici?**
> - **Logistic Regression**: baseline interpretabile, funziona bene con feature lineari
> - **Random Forest**: cattura non-linearità, fornisce feature importance
> - **XGBoost**: state-of-the-art, spesso il migliore in competizioni Kaggle

**Q: Perché XGBoost vince rispetto agli altri modelli?**
> Perché ha la **migliore Accuracy (81.4%)** e la **Precision più alta (0.44)**. Nel contesto F1, predizioni affidabili riducono i falsi allarmi al muretto box, fornendo suggerimenti più precisi per le strategie di gara.

**Q: Perché non usate reti neurali o deep learning?**
> Il dataset è troppo piccolo (~1500 campioni). Le reti neurali richiedono migliaia/milioni di esempi e tendono a overfittare su dataset piccoli. I modelli tradizionali sono più appropriati.

**Q: Avete provato a fare hyperparameter tuning?**
> Usiamo configurazioni standard ottimizzate. Per un dataset di questa dimensione, il tuning aggressivo rischia overfitting. `class_weight='balanced'` e `n_estimators=100` sono scelte robuste.

**Q: Cosa significa eval_metric='logloss' in XGBoost?**
> È la funzione di loss utilizzata per valutare il modello durante il training. Log loss (cross-entropy) è standard per problemi di classificazione binaria.

---

### 📌 Domande sulle Metriche

**Q: Perché non usate solo l'Accuracy?**
> Con dataset sbilanciato, l'accuracy è **fuorviante**. Un modello che predice sempre "no sorpasso" ottiene 92% accuracy ma è completamente inutile (0% recall).

**Q: Cosa rappresenta il F1-Score?**
> È la **media armonica** di Precision e Recall: `F1 = 2 * (P * R) / (P + R)`. Bilancia la capacità di trovare i positivi (recall) con l'affidabilità delle predizioni positive (precision).

**Q: Come si legge la ROC-AUC?**
> Misura la capacità del modello di distinguere tra classi. AUC=0.5 = random classifier, AUC=1.0 = classificatore perfetto. Il nostro 0.75 indica buona capacità discriminativa.

**Q: Perché il Recall è più importante della Precision qui?**
> Nel contesto pit wall, vogliamo **identificare tutte le opportunità di sorpasso**. Qualche falso allarme è accettabile, ma perdere un sorpasso reale può costare posizioni in gara.

---

### 📌 Domande sulla Webapp

**Q: Cosa fa la modalità ATTACK vs DEFENSE?**
> - **ATTACK**: calcola P(il TUO pilota sorpassi l'avversario)
> - **DEFENSE**: calcola P(il tuo pilota VENGA sorpassato)
> Internamente, invertiamo chi è attaccante e chi difensore.

**Q: Come funziona il modello in tempo reale?**
> La webapp è una **simulazione**. L'utente inserisce manualmente i dati. In un contesto reale, i dati verrebbero aggiornati automaticamente giro per giro tramite API F1.

**Q: Perché avete scelto Streamlit?**
> È un framework Python per creare webapp senza JavaScript. Ideale per demo di progetti ML: interfaccia reattiva con poche righe di codice.

---

### 📌 Domande Generali/Teoriche

**Q: Questo modello funzionerebbe su altri circuiti?**
> Le feature relative sono generalizzabili, ma le probabilità assolute potrebbero variare. Circuiti con meno sorpassi (Monaco) darebbero probabilità diverse. Servirebbe fine-tuning o retraining.

**Q: Come migliorereste il modello?**
> - Aggiungere dati da più circuiti con caratteristiche simili a Monza
> - Includere feature aggiuntive (meteo, DRS, posizione in curva)
> - Usare time-series models per catturare la dinamica temporale

**Q: Qual è il limite principale del progetto?**
> Il **dataset limitato** (solo 3 GP, ~1500 campioni). Con più dati potremmo usare modelli più complessi e ottenere predizioni più accurate.

**Q: Come valutereste il modello in produzione?**
> Con **A/B testing**: confrontare le decisioni suggerite dal modello con quelle degli ingegneri reali su gare future, misurando se i sorpassi previsti si verificano.

---

## 📝 Riassunto in 30 Secondi

> "Abbiamo creato un sistema ML che predice i sorpassi in F1 usando dati reali di Monza. Costruiamo feature relative tra piloti (delta tempo, gomme), bilanciamo il dataset sbilanciato con SMOTE, confrontiamo 3 modelli e selezioniamo XGBoost per la miglior Accuracy. La webapp permette di simulare scenari di sorpasso come un vero muretto box."
