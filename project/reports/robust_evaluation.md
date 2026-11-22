### Target `position_gain`

2392 coppie, 403 positivi (prevalenza 0.168). Valutazione leave-one-season-out (Monza 2022, 2023, 2024), predizioni out-of-fold aggregate.

| Modello | Accuracy | F1 | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|
| Baseline: sempre 'no sorpasso' | 0.832 | 0.000 | 0.168 | 0.500 | 0.168 |
| Logistic Regression | 0.687 | 0.401 | 0.451 | 0.723 | 0.197 |
| Random Forest | 0.747 | 0.381 | 0.424 | 0.706 | 0.172 |
| Gradient Boosting | 0.779 | 0.353 | 0.415 | 0.671 | 0.163 |

### Target `on_track_pass`

2161 coppie, 85 positivi (prevalenza 0.039). Valutazione leave-one-season-out (Monza 2022, 2023, 2024), predizioni out-of-fold aggregate.

| Modello | Accuracy | F1 | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|
| Baseline: sempre 'no sorpasso' | 0.961 | 0.000 | 0.039 | 0.500 | 0.039 |
| Logistic Regression | 0.815 | 0.200 | 0.144 | 0.812 | 0.154 |
| Random Forest | 0.898 | 0.185 | 0.143 | 0.832 | 0.072 |
| Gradient Boosting | 0.938 | 0.107 | 0.132 | 0.807 | 0.050 |

