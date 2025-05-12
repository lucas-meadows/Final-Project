import requests
import os
import time
import json
import re
import kagglehub
import pandas as pd
from datetime import datetime, timedelta
from glob import glob


class BettingDataPipeline:
    def __init__(self, api_key, data_dir = "."):
        self.api_key = 'dummy'
        self.base_url = "https://api.the-odds-api.com/v4/historical/sports/mma_mixed_martial_arts/odds"
        self.data_dir = data_dir



    def generate_snapshot_times(self, start, end, interval_minutes=30):
        current = start
        times = []
        while current <= end:
            times.append(current.isoformat() + "Z")  
            current += timedelta(minutes=interval_minutes)
        return times



    def get_snapshot(self, iso_timestamp):
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'american',
            'date': iso_timestamp
        }
        res = requests.get(self.base_url, params=params)
        return res.json() if res.status_code ==200 else None



    def download_snapshots(self, start, end, interval_minutes=30):
        times = self.generate_snapshot_times(start, end, interval_minutes)
        for i, ts in enumerate(times):
            print(f"Downloading snapshot {i+1}/{len(snapshots)} — {ts}")
            data = get_snapshot(ts)
            if data:
                filename = f'ufc_snapshot_{i}_{ts.replace(":", "-")}.json'
                with open(filename, 'w') as f:
                    json.dump(data, f)
            time.sleep(0.01)  # to keep within API rate limits



    def implied_prob(self, price):
        try:
            price = float(price)
            if price > 0:
                return 100 / (price + 100)
            else:
                return -price / (-price + 100)
        except:
            return None



    def extract_odds_from_snapshot(self, file_path):
        with open(file_path) as f:
            data = json.load(f)

        if "data" not in data or not data["data"]:
            return pd.DataFrame()  # skip empty snapshots

        records = []
        snapshot_id = data.get("snapshot", os.path.basename(file_path))
        timestamp = data.get("timestamp")

        for fight in data["data"]:
            fight_id = fight.get("id")
            home = fight.get("home_team")
            away = fight.get("away_team")
            if "UFC" not in fight.get("sport_title", "") and "ufc" not in fight.get("sport_title", "").lower():
                continue

            for bookmaker in fight.get("bookmakers", []):
                bm = bookmaker.get("title", "unknown")
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "h2h":
                        for outcome in market.get("outcomes", []):
                            fighter = outcome.get("name")
                            price = outcome.get("price")
                            opponent = away if fighter == home else home
                            prob = self.implied_prob(price)
                            records.append({
                                "snapshot_id": snapshot_id,
                                "timestamp": timestamp,
                                "fight_id": fight_id,
                                "fighter": fighter,
                                "opponent": opponent,
                                "price": price,
                                "implied_prob": prob,
                                "bookmaker": bm
                            })
        return pd.DataFrame(records)

    def get_snapshot_number(self, filename):
        match = re.search(r'ufc_snapshot_(\d+)_', filename)
        return int(match.group(1)) if match else -1

    def load_all_snapshots(self):
        all_files = sorted(glob(os.path.join(self.data_dir, "ufc_snapshot_*.json")), key=self.get_snapshot_number)
        dfs = []
        for file in all_files:
            print(f"📦 Processing: {os.path.basename(file)}")
            try:
                df = self.extract_odds_from_snapshot(file)
                if not df.empty:
                    dfs.append(df)
                else:
                    print(f"Empty or skipped: {os.path.basename(file)}")
            except Exception as e:
                print(f"Error in {os.path.basename(file)}: {e}")

        if dfs:
            full_df = pd.concat(dfs, ignore_index=True)
            full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
            full_df = full_df.sort_values(by=['fight_id', 'fighter', 'timestamp'])
            print(f"\nCombined DataFrame shape: {full_df.shape}")
            print(full_df.head())
            return full_df
        else:
            print("No valid data extracted.")
            return pd.DataFrame()




# Importing Kaggle Dataset for the static fighter data

def load_kaggle():# Download latest version
    path = kagglehub.dataset_download("tawhidmonowar/ufc-fighters-stats-and-records-dataset")

    file_path = os.path.join(path, "ufc_fighters_stats_and_records.json")

    df = pd.read_json(file_path)
    return df



# Runs this data_pipeline in the terminal with our API key and desired start and end dates, to download the data
if __name__ == "__main__":
    api_key = 'ccf60e55f93cc55e91f721466b474517'
    pipeline = BettingDataPipeline(api_key=api_key, data_dir="market_snapshots")

    start = datetime(2023, 11, 1)
    end = datetime(2023, 11, 30)

    df = pipeline.load_all_snapshots()
    print("Finished loading all snapshot data.")
