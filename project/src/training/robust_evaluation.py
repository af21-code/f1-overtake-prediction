"""
================================================================================
ROBUST EVALUATION - F1 Overtake Prediction
================================================================================
Valutazione piu rigorosa del modello, complementare a model_trainer.py.

Cosa cambia rispetto alla valutazione originale:
1. Baseline: confronto con un classificatore che predice sempre "nessun
   sorpasso". Con ~17% di positivi questa baseline ha gia ~83% di accuracy,
   quindi l'accuracy da sola non dice nulla.
2. Metriche adatte a classi sbilanciate: PR-AUC (average precision), ROC-AUC,
   F1, Brier score. Il riferimento per la PR-AUC e la prevalenza dei positivi.
3. Split per stagione (Leave-One-Season-Out): si addestra su due GP di Monza
   e si testa sul terzo. Lo split casuale originale mette giri della stessa
   gara sia nel train sia nel test, e giri consecutivi sono molto correlati.
4. Scaling dentro la pipeline (stimato solo sul fold di train). Lo
   sbilanciamento e gestito con class_weight='balanced' invece di SMOTE, cosi
   lo script usa solo scikit-learn e i numeri sono riproducibili ovunque.
   Il gradient boosting e HistGradientBoostingClassifier di scikit-learn,
   lo stesso tipo di modello di XGBoost.
5. Due definizioni del target:
   - position_gain: il target originale (l'attaccante guadagna una posizione
     nel giro successivo, per qualunque motivo);
   - on_track_pass: l'attaccante e davanti al difensore nel giro successivo e
     nessuno dei due si e fermato ai box nei due giri successivi. Le coppie
     con un pit stop vengono escluse: non sono duelli in pista.

Uso:
    cd project/src
    python training/robust_evaluation.py

Output:
    reports/robust_evaluation.json
    reports/robust_evaluation.md
    reports/pr_curves_<target>.png
================================================================================
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW_PATH = os.path.join(PROJECT, "data", "f1_ground_effect_dataset.csv")
REL_PATH = os.path.join(PROJECT, "data", "f1_monza_relative_features.csv")
REPORTS = os.path.join(PROJECT, "reports")

FEATURES = [
    "Attacker_Position",
    "Delta_LapTime",
    "Delta_TyreLife",
    "Compound_Advantage",
    "Attacker_LapTime",
    "Attacker_TyreLife",
]
SEED = 42


# ------------------------------------------------------------------------------
# Target
# ------------------------------------------------------------------------------
def add_on_track_label(rel: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge PitInvolved e OnTrackPass a ogni coppia attaccante/difensore."""
    idx = raw.set_index(["Year", "LapNumber", "Driver"])[["Position", "TyreLife"]]
    lookup = idx.to_dict("index")

    def lap(year, lap_no, driver):
        return lookup.get((year, lap_no, driver))

    pit_involved, passed = [], []
    for r in rel.itertuples(index=False):
        pit = False
        for drv, tyre_life in (
            (r.Attacker_Driver, r.Attacker_TyreLife),
            (r.Defender_Driver, r.Defender_TyreLife),
        ):
            for k in (1, 2):
                nxt = lap(r.Year, r.LapNumber + k, drv)
                # giro mancante o vita gomma azzerata = pit stop / ritiro
                if nxt is None or nxt["TyreLife"] < tyre_life:
                    pit = True
        att = lap(r.Year, r.LapNumber + 1, r.Attacker_Driver)
        dfn = lap(r.Year, r.LapNumber + 1, r.Defender_Driver)
        ok = att is not None and dfn is not None
        pit_involved.append(pit)
        passed.append(int(ok and att["Position"] < dfn["Position"]))

    out = rel.copy()
    out["PitInvolved"] = pit_involved
    out["OnTrackPass"] = passed
    return out


# ------------------------------------------------------------------------------
# Modelli
# ------------------------------------------------------------------------------
def build_models():
    def pipe(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    return {
        "Baseline: sempre 'no sorpasso'": pipe(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": pipe(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        ),
        "Random Forest": pipe(
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                random_state=SEED, n_jobs=-1,
            )
        ),
        "Gradient Boosting": pipe(
            HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced")
        ),
    }


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    single_class = len(np.unique(y)) < 2
    return {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": float("nan") if single_class else average_precision_score(y, prob),
        "roc_auc": float("nan") if single_class else roc_auc_score(y, prob),
        "brier": brier_score_loss(y, prob),
    }


def leave_one_season_out(df: pd.DataFrame, target: str):
    X = df[FEATURES].fillna(0).to_numpy()
    y = df[target].to_numpy()
    years = df["Year"].to_numpy()
    models = build_models()

    results = {}
    for name, model in models.items():
        oof = np.zeros(len(y))
        per_season = {}
        for season in sorted(np.unique(years)):
            tr, te = years != season, years == season
            model.fit(X[tr], y[tr])
            oof[te] = model.predict_proba(X[te])[:, 1]
            per_season[int(season)] = metrics(y[te], oof[te])
        results[name] = {
            "pooled": metrics(y, oof),
            "per_season": per_season,
            "oof": oof,
        }
    return results, y


def plot_pr(results, y, target):
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        if name.startswith("Baseline"):
            continue
        p, r, _ = precision_recall_curve(y, res["oof"])
        plt.plot(r, p, linewidth=2,
                 label=f"{name} (PR-AUC {res['pooled']['pr_auc']:.3f})")
    plt.axhline(y.mean(), color="k", linestyle="--",
                label=f"Prevalenza positivi ({y.mean():.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall, leave-one-season-out - target: {target}")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS, f"pr_curves_{target}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def to_markdown(summary):
    lines = []
    for target, block in summary["targets"].items():
        lines.append(f"### Target `{target}`")
        lines.append("")
        lines.append(
            f"{block['n_samples']} coppie, {block['n_positive']} positivi "
            f"(prevalenza {block['prevalence']:.3f}). "
            "Valutazione leave-one-season-out (Monza 2022, 2023, 2024), "
            "predizioni out-of-fold aggregate."
        )
        lines.append("")
        lines.append("| Modello | Accuracy | F1 | PR-AUC | ROC-AUC | Brier |")
        lines.append("|---|---|---|---|---|---|")
        for name, m in block["models"].items():
            p = m["pooled"]
            lines.append(
                f"| {name} | {p['accuracy']:.3f} | {p['f1']:.3f} | "
                f"{p['pr_auc']:.3f} | {p['roc_auc']:.3f} | {p['brier']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    rel = pd.read_csv(REL_PATH)
    rel = add_on_track_label(rel, raw)

    pos = rel["IsOvertake"] == 1
    label_audit = {
        "position_gain_positives": int(pos.sum()),
        "of_which_pit_involved": int((pos & rel["PitInvolved"]).sum()),
        "of_which_passed_defender_without_pit": int(
            (pos & ~rel["PitInvolved"] & (rel["OnTrackPass"] == 1)).sum()
        ),
    }

    datasets = {
        "position_gain": rel.assign(target=rel["IsOvertake"]),
        "on_track_pass": rel[~rel["PitInvolved"]].assign(
            target=rel.loc[~rel["PitInvolved"], "OnTrackPass"]
        ),
    }

    summary = {
        "label_audit": label_audit,
        "targets": {},
    }
    for target, df in datasets.items():
        results, y = leave_one_season_out(df, "target")
        plot_pr(results, y, target)
        summary["targets"][target] = {
            "n_samples": int(len(y)),
            "n_positive": int(y.sum()),
            "prevalence": float(y.mean()),
            "models": {
                n: {"pooled": r["pooled"], "per_season": r["per_season"]}
                for n, r in results.items()
            },
        }

    with open(os.path.join(REPORTS, "robust_evaluation.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md = to_markdown(summary)
    with open(os.path.join(REPORTS, "robust_evaluation.md"), "w") as f:
        f.write(md + "\n")

    print("Label audit:", json.dumps(label_audit, indent=2))
    print()
    print(md)


if __name__ == "__main__":
    main()
