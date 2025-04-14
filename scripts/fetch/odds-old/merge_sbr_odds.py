import os
import json
import pandas as pd
from datetime import datetime

# Constants
DATA_DIR = "c:/projects/nba_prediction/data/raw/betting"
DAILY_DIR = os.path.join(DATA_DIR, "sbr_daily")
MASTER_FILE = os.path.join(DATA_DIR, "nba_sbr_odds_2025.csv")

# Game periods and odds types
GAME_PERIODS = [
    'full-game', '1st-half', '2nd-half',
    '1st-quarter', '2nd-quarter', '3rd-quarter', '4th-quarter'
]

ODDS_TYPES = ['spreads', 'moneylines', 'totals']
PERIOD_SCORE_KEYS = {
    '1st-quarter': 1,
    '2nd-quarter': 2,
    '3rd-quarter': 3,
    '4th-quarter': 4,
    '1st-half': 1,
    '2nd-half': 2
}

def extract_scores(score_data_list):
    scores = {}
    for score in score_data_list:
        period = score.get("Period")
        is_home = score.get("isHomeTeam")
        points = score.get("Points", 0)
        if period is not None:
            label = f"{period}_period_{'home' if is_home else 'away'}Points"
            scores[label] = points
    return scores

def extract_odds_data_by_bookmaker(odds_views, odds_type, period):
    results = {}
    for view in odds_views:
        if not view:
            continue
        book = view.get("sportsbook", "unknown").lower().replace(" ", "")

        for timing in ["openingLine", "currentLine"]:
            line = view.get(timing, {})
            suffix = "opening" if timing == "openingLine" else "current"

            if odds_type == "moneylines":
                for side in ["home", "away"]:
                    col = f"{period}_moneylineOdds_{side}_{book}_{suffix}"
                    results[col] = line.get(f"{side}Odds")

            elif odds_type == "spreads":
                for side in ["home", "away"]:
                    spread_col = f"{period}_spread_{side}_{book}_{suffix}"
                    odds_col = f"{period}_spreadOdds_{side}_{book}_{suffix}"
                    results[spread_col] = line.get(f"{side}Spread")
                    results[odds_col] = line.get(f"{side}Odds")

            elif odds_type == "totals":
                for direction in ["over", "under"]:
                    col = f"{period}_totalsOdds_{direction}_{book}_{suffix}"
                    results[col] = line.get(f"{direction}Odds")
                total_col = f"{period}_totals_{book}_{suffix}"
                results[total_col] = line.get("total")

    return results

def process_game_data(data, date_str):
    games = {}
    for odds_type in ODDS_TYPES:
        for period in GAME_PERIODS:
            odds_tables = data.get(odds_type, {}).get(period, {}).get('pageProps', {}).get('oddsTables', [])
            if not odds_tables:
                continue
            game_rows = odds_tables[0].get('oddsTableModel', {}).get('gameRows', [])

            for game in game_rows:
                game_view = game.get('gameView', {})
                game_id = game_view.get('gameId')
                if not game_id:
                    continue

                if game_id not in games:
                    base = {
                        'game_id': game_id,
                        'date': date_str,
                        'home_team': game_view.get('homeTeam', {}).get('fullName'),
                        'away_team': game_view.get('awayTeam', {}).get('fullName'),
                        'home_team_score': game_view.get('homeTeamScore'),
                        'away_team_score': game_view.get('awayTeamScore'),
                        'game_status': game_view.get('gameStatusText'),
                        'start_time': game_view.get('startDate'),
                        'venue': game_view.get('venueName'),
                        'city': game_view.get('city'),
                        'state': game_view.get('state')
                    }
                    # Pull scores
                    score_data = game.get("liveScoreViews", {}).get("viewdata", {}).get("GameTeamScoreDataList", [])
                    base.update(extract_scores(score_data))
                    games[game_id] = base

                odds_data = extract_odds_data_by_bookmaker(game.get("oddsViews", []), odds_type, period)
                games[game_id].update(odds_data)

    return list(games.values())

def main():
    print("🔄 Starting merge of SBR odds with expanded format...")
    json_files = sorted([f for f in os.listdir(DAILY_DIR) if f.endswith('.json')])
    all_records = []

    for file in json_files:
        date_str = file.replace(".json", "")
        print(f"📅 Processing {date_str}...")
        try:
            with open(os.path.join(DAILY_DIR, file), "r") as f:
                data = json.load(f)
            records = process_game_data(data, date_str)
            all_records.extend(records)
            print(f"✅ Added {len(records)} records")
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")

    if all_records:
        print(f"💾 Saving merged data to {MASTER_FILE}...")
        df = pd.DataFrame(all_records)
        df.to_csv(MASTER_FILE, index=False)
        print("✅ Merge complete.")
    else:
        print("⚠️ No records to save!")

if __name__ == "__main__":
    main()
