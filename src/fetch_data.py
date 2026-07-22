import fastf1 
import pandas as pd
import os

os.makedirs('data/raw/cache', exist_ok=True)
fastf1.Cache.enable_cache('data/raw/cache')

def fetch_season(year):
    schedule = fastf1.get_event_schedule(year)
    
    rows = []
    
    for event in schedule.itertuples():
        if event.EventFormat == 'testing': #skip testing rounds
            continue
        try:
            session = fastf1.get_session(year, event.RoundNumber, 'R')
            session.load(telemetry=False, laps=False, weather=True)
            results = session.results.copy()
            results['season'] = year
            results['round'] = event.RoundNumber
            results['circuit'] = event.EventName
            weather = session.weather_data
            
            results['avg_air_temp'] = weather['AirTemp'].mean()
            results['avg_track_temp'] = weather['TrackTemp'].mean()
            
            results['rain'] = weather['Rainfall'].any()
            rows.append(results)
        except Exception as e:
            print(f"Skipping {event.EventName}: {e}")
            
        
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    for year in [2022, 2023, 2024, 2025, 2026]:
        df = fetch_season(year)
        df.to_csv(f'data/raw/season_{year}.csv', index=False)