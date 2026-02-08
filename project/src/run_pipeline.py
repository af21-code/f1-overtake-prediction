#!/usr/bin/env python
"""
================================================================================
RUN PIPELINE - F1 Overtake Prediction
================================================================================
Script per eseguire l'intera pipeline di training in sequenza.

Esegue automaticamente:
1. Download dati FastF1 (Monza 2022-2024)
2. Costruzione feature relative
3. Preprocessing e bilanciamento SMOTE
4. Training e valutazione modelli

Utilizzo:
    cd project/src
    python run_pipeline.py

Output:
    - data/f1_ground_effect_dataset.csv
    - data/f1_monza_relative_features.csv
    - data/processed/*.npy
    - models/best_model.pkl
    - reports/*.png, *.json
================================================================================
"""

import subprocess
import sys
import os

# Colori per output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'


def run_step(step_num, description, script_path):
    """Esegue uno step della pipeline."""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{YELLOW}[STEP {step_num}]{RESET} {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode != 0:
        print(f"{RED}[ERRORE] Step {step_num} fallito!{RESET}")
        return False
    
    print(f"{GREEN}[OK] Step {step_num} completato{RESET}")
    return True


def main():
    print(f"""
{BOLD}╔══════════════════════════════════════════════════════════╗
║     F1 OVERTAKE PREDICTION - TRAINING PIPELINE           ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")
    
    steps = [
        ("1", "Download dati FastF1", "pipeline/data_loader.py"),
        ("2", "Costruzione feature relative", "pipeline/relative_feature_builder.py"),
        ("3", "Preprocessing e SMOTE", "pipeline/feature_processor.py"),
        ("4", "Training modelli", "training/model_trainer.py"),
    ]
    
    for step_num, description, script in steps:
        if not run_step(step_num, description, script):
            print(f"\n{RED}Pipeline interrotta a causa di un errore.{RESET}")
            sys.exit(1)
    
    print(f"""
{GREEN}{BOLD}╔══════════════════════════════════════════════════════════╗
║              PIPELINE COMPLETATA CON SUCCESSO!           ║
╚══════════════════════════════════════════════════════════╝{RESET}

{BOLD}Output generati:{RESET}
  • data/f1_ground_effect_dataset.csv
  • data/f1_monza_relative_features.csv
  • models/best_model.pkl
  • reports/training_report.json

{BOLD}Per avviare la webapp:{RESET}
  cd app
  streamlit run app.py
""")


if __name__ == "__main__":
    main()
