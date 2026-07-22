from pathlib import Path
import numpy as np
import pandas as pd

raw_dir = Path('data/raw')
processed_dir = Path('data/processed')

finished_status = {'Finished'}

def is_finished(status):
    if pd.isna(status):
        return False
    status = str(status)
    if status in finished_status:
        return True
    if status.startswith('+'):
        return True
    return False

def load_raw():
    files = sorted(raw_dir.glob('season_*.csv'))
    
    if not files:
        raise FileNotFoundError('No season_*.csv files found in data/raw/. Run fetch_data.py first.')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df

def build_features(df):
    df = df.copy()
    finished_mask = (
        df['Status'].str.startswith(('Finished', '+', 'Lapped'), na=False) 
        | (df['Status'] == 'Lapped')
    )
    df['is_dnf'] = (~finished_mask).astype(int)#########1= DNF 0= finished
    
    # Sort chronologically by season and round
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df.sort_values(["event_date", "round"]).reset_index(drop=True)
    else:
        df = df.sort_values(["season", "round"]).reset_index(drop=True)
    
    driver_key = "DriverId" if "DriverId" in df.columns else "Abbreviation"
    
    
    df["driver_dnf_rate_last5"] = (#### driver dnf rate last 5 races EXCLUDING(shift(1) func) current race
        df.groupby(driver_key)['is_dnf'].transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
    )
    
    df["team_dnf_rate_last5"] = (#### team dnf rate last 5 races EXCLUDING(shift(1) func) current race
        df.groupby('TeamName')['is_dnf'].transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
    )
    
    df["circuit_dnf_rate_hist"] = (#### circuite dnf rate for all races EXCLUDING(shift(1) func) current race
        df.groupby("circuit")["is_dnf"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    
    
    global_dnf_rate = df["is_dnf"].mean()#baseline to handle if there are new drivers or new circuits
    for col in ["driver_dnf_rate_last5", "team_dnf_rate_last5", "circuit_dnf_rate_hist"]:
        df[col] = df[col].fillna(global_dnf_rate)
        
    feature_cols = [
        'season', 'round', 'circuit', 'event_date', driver_key, 'Abbreviation', 'TeamName', 'GridPosition',
        'avg_air_temp', 'avg_track_temp', "rain", "driver_dnf_rate_last5", "team_dnf_rate_last5",
        "circuit_dnf_rate_hist", "is_dnf"
    ]
    
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    return df[feature_cols]


def main():
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    
    raw = load_raw()
    
    print(f"Loaded {len(raw)} raw rows across {raw['season'].nunique()} seasons")
    features = build_features(raw)
    
    out_path = processed_dir / 'features.csv'
    features.to_csv(out_path, index=False)
    
    print(f"Saved {len(features)} rows, {features.shape[1]} columns to {out_path}")
    print("\nDNF rate overall:", round(features["is_dnf"].mean(), 3))
    print("\nSample:")
    print(features.head())
    
    
if __name__ == "__main__":
    main()