from data_pipeline import BettingDataPipeline, load_kaggle
import pandas as pd
import numpy as np

pipeline = BettingDataPipeline(api_key="dummy", data_dir="market_snapshots")
odds_df = pipeline.load_all_snapshots()

fighter_df = load_kaggle()

odds_df.to_csv("combined_odds_data.csv", index=False)
fighter_df.to_csv("kaggle_fighter_stats.csv", index=False)

# Function to extract features from our data

def extract_basic_features(df):
    features = []


    for (fight_id, fighter), group in odds_df.groupby(['fight_id', 'fighter']):
        group_sorted = group.sort_values('timestamp')
        series = group_sorted['implied_prob'].dropna()

        # Skip for too few data points
        if len(series) < 2:
            continue

        start_prob = series.iloc[0]
        end_prob = series.iloc[-1]
        prob_change = end_prob - start_prob
        num_changes = (series.diff().fillna(0) != 0).sum()
        large_move = abs(prob_change) > 0.02

        # Cross-book spread per timestamp
        spread_per_time = group_sorted.groupby('timestamp')['implied_prob'].agg(lambda x: x.max() - x.min())
        avg_spread = spread_per_time.mean()

        bookmaker_count = group_sorted.groupby('timestamp')['bookmaker'].nunique().mean()

        time_range_hours = (group_sorted['timestamp'].max() - group_sorted['timestamp'].min()).total_seconds() / 3600

        features.append({
            'fight_id': fight_id,
            'fighter': fighter,
            'start_prob': start_prob,
            'end_prob': end_prob,
            'prob_change': prob_change,
            'num_price_changes': num_changes,
            'avg_spread_across_books': avg_spread,
            'avg_num_books': bookmaker_count,
            'time_to_fight_hrs': time_range_hours,
            'large_movement': int(large_move),
        })

    return pd.DataFrame(features)

df_features = extract_basic_features(odds_df)

df_features.to_csv("engineered_fight_features.csv", index=False)

print(f"Engineered features for {len(df_features)} fight-fighter combos")
print(df_features.head())