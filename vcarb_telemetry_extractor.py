import os
import fastf1
import pandas as pd

def extract_vcarb_battle_features(year: int = 2024, gp: str = "Imola", session_type: str = "R"):
    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache("cache")

    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False)

    # Filtra i giri del team Visa Cash App RB
    vcarb_laps = session.laps.pick_teams("RB").pick_quicklaps()

    records = []
    for _, lap in vcarb_laps.iterrows():
        tel = lap.get_car_data().add_distance()
        records.append({
            "driver": lap["Driver"],
            "lap_number": int(lap["LapNumber"]),
            "lap_time_s": lap["LapTime"].total_seconds(),
            "compound": lap["Compound"],
            "tyre_life": int(lap["TyreLife"]),
            "top_speed_kph": float(tel["Speed"].max()),
            "full_throttle_pct": round((tel["Throttle"] >= 98).mean() * 100, 2),
            "drs_active_pct": round((tel["DRS"] >= 10).mean() * 100, 2)
        })

    df = pd.DataFrame(records)
    df.to_csv("vcarb_imola_telemetry_features.csv", index=False)
    print(f"Extracted {len(df)} clean laps for Visa Cash App RB at {gp} {year}.")
    return df

if __name__ == "__main__":
    extract_vcarb_battle_features()
