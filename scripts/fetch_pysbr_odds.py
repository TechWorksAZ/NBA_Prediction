from pysbr import NBA, EventsByDate, Sportsbooks, LineHistory
from datetime import datetime, timedelta
import pandas as pd
import os

# --- CONFIG ---
START_DATE = datetime(2024, 10, 22)
END_DATE = datetime(2025, 4, 10)
SAVE_PATH = "C:/projects/nba_prediction/data/raw/betting/pysbr_nba_odds.csv"
SPORT = NBA()

# Define static market IDs via pysbr (manually mapped)
MARKET_IDS = {
    "full-game_spread": 401,
    "full-game_moneyline": 402,
    "full-game_total": 403,
    "1st-half_spread": 408,
    "1st-half_moneyline": 409,
    "1st-half_total": 410,
    "2nd-half_spread": 411,
    "2nd-half_moneyline": 412,
    "2nd-half_total": 413,
    "1st-quarter_spread": 414,
    "1st-quarter_moneyline": 415,
    "1st-quarter_total": 416,
    "2nd-quarter_spread": 417,
    "2nd-quarter_moneyline": 418,
    "2nd-quarter_total": 419,
    "3rd-quarter_spread": 420,
    "3rd-quarter_moneyline": 421,
    "3rd-quarter_total": 422,
    "4th-quarter_spread": 423,
    "4th-quarter_moneyline": 424,
    "4th-quarter_total": 425
}

nba = NBA()
all_sportsbook_ids = [s["id"] for s in Sportsbooks(nba.league_id).list()]
sportsbooks = Sportsbooks(all_sportsbook_ids)

all_games = []
date = START_DATE
while date <= END_DATE:
    print(f"\n📅 Fetching games for {date.date()}...")
    events = EventsByDate(nba.league_id, date).list()
    for event in events:
        event_id = event["eventId"]
        base_game_data = {
            'event_id': event_id,
            'game_date': event["startDate"][:10],
            'home_team': event["homeTeam"],
            'away_team': event["awayTeam"]
        }

        # Loop through each sportsbook ID
        for sbook in sportsbooks.list():
            book_id = sbook["id"]
            book_name = sbook["name"].lower().replace(" ", "")
            game_data = base_game_data.copy()
            game_data["sportsbook"] = book_name

            for label, market_id in MARKET_IDS.items():
                try:
                    lh = LineHistory(event_id, market_id, book_id)
                    df = lh.dataframe()
                    if df is not None and not df.empty:
                        latest = df.sort_values("createdAt", ascending=False).iloc[0]
                        for col in ["homeOdds", "awayOdds", "homeSpread", "awaySpread", "total", "overOdds", "underOdds"]:
                            if col in latest:
                                game_data[f"{label}_{col}"] = latest[col]
                except Exception as e:
                    print(f"⚠️ Error fetching {label} from {book_name} for {event['homeTeam']} vs {event['awayTeam']} on {date.date()}: {e}")

            all_games.append(game_data)
    date += timedelta(days=1)

# Combine and save
df_all = pd.DataFrame(all_games)
df_all.to_csv(SAVE_PATH, index=False)
print(f"\n✅ Done. Saved 1-row-per-game odds data to {SAVE_PATH}")
