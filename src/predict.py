
import argparse
from pathlib import Path

import fastf1
import joblib
import pandas as pd

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
MODELS_DIR = Path("models")

NUMERIC_FEATURES = [
    "GridPosition",
    "avg_air_temp",
    "avg_track_temp",
    "driver_dnf_rate_last5",
    "team_dnf_rate_last5",
    "circuit_dnf_rate_hist",
]
CATEGORICAL_FEATURES = ["TeamName", "rain"]


def get_qualifying_grid(year: int, event: str) -> pd.DataFrame:
    """Pull actual grid positions + teams from the qualifying session."""
    fastf1.Cache.enable_cache(str(RAW_DIR / "cache"))
    session = fastf1.get_session(year, event, "Q")
    session.load(laps=False, telemetry=False, weather=False, messages=False)

    results = session.results.copy()
    driver_key = "DriverId" if "DriverId" in results.columns else "Abbreviation"

    grid = results[[driver_key, "Abbreviation", "TeamName", "Position"]].rename(
        columns={"Position": "GridPosition"}
    )
    return grid, driver_key


def get_latest_reliability(features_df: pd.DataFrame, driver_key: str) -> pd.DataFrame:
    """For each driver/team, take their most recent rolling DNF rate
    from the historical feature table - i.e. their reliability figure
    going into the next race."""
    latest_per_driver = (
        features_df.sort_values(["season", "round"])
        .groupby(driver_key)
        .tail(1)[[driver_key, "driver_dnf_rate_last5"]]
    )
    latest_per_team = (
        features_df.sort_values(["season", "round"])
        .groupby("TeamName")
        .tail(1)[["TeamName", "team_dnf_rate_last5"]]
    )
    return latest_per_driver, latest_per_team


def get_circuit_history(features_df: pd.DataFrame, event: str) -> float:
    """Historical DNF rate at this circuit across all prior races.
    Falls back to the global average if the circuit is new / not matched.
    """
    matches = features_df[features_df["circuit"].str.contains(event.split(" ")[0], case=False, na=False)]
    if matches.empty:
        return features_df["is_dnf"].mean()
    return matches["is_dnf"].mean()


def get_circuit_weather_estimate(features_df: pd.DataFrame, event: str):
    matches = features_df[features_df["circuit"].str.contains(event.split(" ")[0], case=False, na=False)]
    if matches.empty:
        return features_df["avg_air_temp"].mean(), features_df["avg_track_temp"].mean(), False
    rain_history = matches["rain"].mean() if "rain" in matches else 0
    return matches["avg_air_temp"].mean(), matches["avg_track_temp"].mean(), rain_history > 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--event", type=str, default="Hungarian Grand Prix")
    args = parser.parse_args()

    features_df = pd.read_csv(PROCESSED_DIR / "features.csv")
    pipeline = joblib.load(MODELS_DIR / "dnf_model.pkl")

    grid, driver_key = get_qualifying_grid(args.year, args.event)
    latest_driver, latest_team = get_latest_reliability(features_df, driver_key)
    circuit_dnf_rate = get_circuit_history(features_df, args.event)
    air_temp, track_temp, rain_guess = get_circuit_weather_estimate(features_df, args.event)

    df = grid.merge(latest_driver, on=driver_key, how="left")
    df = df.merge(latest_team, on="TeamName", how="left")

    df["circuit_dnf_rate_hist"] = circuit_dnf_rate
    df["avg_air_temp"] = air_temp
    df["avg_track_temp"] = track_temp
    df["rain"] = rain_guess

    global_dnf_rate = features_df["is_dnf"].mean()
    df["driver_dnf_rate_last5"] = df["driver_dnf_rate_last5"].fillna(global_dnf_rate)
    df["team_dnf_rate_last5"] = df["team_dnf_rate_last5"].fillna(global_dnf_rate)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    df["dnf_probability"] = pipeline.predict_proba(X)[:, 1]

    output = df[["Abbreviation", "TeamName", "GridPosition", "dnf_probability"]].sort_values(
        "dnf_probability", ascending=False
    )
    output["dnf_probability"] = (output["dnf_probability"] * 100).round(1)

    print(f"\nDNF risk predictions - {args.event} {args.year}\n")
    print(output.to_string(index=False))

    out_path = PROCESSED_DIR / f"prediction_{args.year}_{args.event.replace(' ', '_')}.csv"
    output.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print("\nNote: weather is a historical circuit average, not a live forecast -")
    print("swap in a real forecast API for a more accurate pre-race estimate.")


if __name__ == "__main__":
    main()