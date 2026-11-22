"""
================================================================================
TEAM LAP FEATURES - F1 Overtake Prediction
================================================================================
Estrae, per i giri puliti di un team in una sessione, le feature di telemetria
utili a valutare il potenziale di attacco in rettilineo:

- top_speed_kph:      velocita massima sul giro
- full_throttle_pct:  percentuale del giro a gas pieno (throttle >= 98%)
- drs_active_pct:     percentuale del giro con DRS aperto (codici FastF1 >= 10)

Sono le grandezze che il dataset di Monza non contiene (usa solo tempi sul
giro, gomme e posizioni) e che servirebbero per estendere il modello di
sorpasso con informazioni di telemetria vera.

Uso (dalla cartella project/src):
    python analysis/team_lap_features.py --year 2024 --gp Imola --team RB

Output: data/team_lap_features_<team>_<gp>_<year>.csv
================================================================================
"""

import argparse
import os

import fastf1
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE_DIR = os.path.join(PROJECT, "cache")
DATA_DIR = os.path.join(PROJECT, "data")


def extract_team_lap_features(year: int, gp: str, team: str, session_type: str = "R") -> pd.DataFrame:
    """Restituisce una riga per ogni giro pulito dei piloti del team."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False)

    # Solo giri rappresentativi: niente giri di uscita/rientro box o anomali
    laps = session.laps.pick_teams(team).pick_quicklaps()

    records = []
    for _, lap in laps.iterrows():
        tel = lap.get_car_data().add_distance()
        records.append({
            "driver": lap["Driver"],
            "lap_number": int(lap["LapNumber"]),
            "lap_time_s": lap["LapTime"].total_seconds(),
            "compound": lap["Compound"],
            "tyre_life": int(lap["TyreLife"]),
            "top_speed_kph": float(tel["Speed"].max()),
            "full_throttle_pct": round((tel["Throttle"] >= 98).mean() * 100, 2),
            "drs_active_pct": round((tel["DRS"] >= 10).mean() * 100, 2),
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Feature di telemetria per i giri di un team.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--gp", default="Imola")
    parser.add_argument("--team", default="RB", help="Nome del team come in FastF1 (es. RB, Ferrari)")
    parser.add_argument("--session", default="R", help="Tipo di sessione (R, Q, FP1...)")
    args = parser.parse_args()

    df = extract_team_lap_features(args.year, args.gp, args.team, args.session)
    os.makedirs(DATA_DIR, exist_ok=True)
    name = f"team_lap_features_{args.team}_{args.gp}_{args.year}.csv".replace(" ", "_")
    out = os.path.join(DATA_DIR, name)
    df.to_csv(out, index=False)
    print(f"[OK] {len(df)} giri puliti di {args.team} a {args.gp} {args.year} -> {out}")


if __name__ == "__main__":
    main()
