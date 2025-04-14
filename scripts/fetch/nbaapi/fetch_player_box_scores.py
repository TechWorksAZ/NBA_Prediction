import os
import time
import pandas as pd
import requests
from nba_api.stats.endpoints import LeagueGameFinder, BoxScoreTraditionalV2

# === CONFIG ===
SEASON = "2024-25"
REQUEST_DELAY = 0.75
COOLDOWN_EVERY = 50
COOLDOWN_TIME = 10
RETRIES = 3

# === PATHS ===
RAW_DATA_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, "player_box_scores_traditional_2025.csv")

# === LOAD EXISTING ===
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE, dtype={"GAME_ID": str})
    processed_ids = set(existing_df["GAME_ID"].astype(str).unique())
    all_data = [existing_df]
    print(f"✅ Loaded {len(existing_df)} box score rows for {len(processed_ids)} games.")
else:
    processed_ids = set()
    all_data = []

# === GET GAMES TO FETCH ===
print("📥 Fetching list of NBA games...")
finder = LeagueGameFinder(season_nullable=SEASON, season_type_nullable="Regular Season")
games_df = finder.get_data_frames()[0]
games_df = games_df[(games_df["TEAM_ID"] >= 1610612737) & (games_df["TEAM_ID"] <= 1610612766)]
all_game_ids = games_df["GAME_ID"].astype(str).unique()
unprocessed_ids = [gid for gid in all_game_ids if gid not in processed_ids]
print(f"🔄 Already processed: {len(processed_ids)} | 🆕 Remaining: {len(unprocessed_ids)}")

# === FETCH FUNCTION ===
def fetch_boxscore(game_id, retries=RETRIES):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nba.com/",
        "Accept": "application/json",
        "Connection": "keep-alive"
    }

    for attempt in range(1, retries + 1):
        try:
            result = BoxScoreTraditionalV2(game_id=game_id, headers=headers)
            df = result.get_data_frames()[0]
            if df.empty:
                print(f"⚠️ Empty result for {game_id}")
                return None
            return df
        except requests.exceptions.ReadTimeout:
            print(f"⏳ Timeout for {game_id} (attempt {attempt})")
            time.sleep(2 + attempt)
        except Exception as e:
            print(f"❌ Failed to fetch {game_id}: {e}")
            return None
    return None

# === MAIN LOOP ===
for i, game_id in enumerate(unprocessed_ids, 1):
    print(f"\n[{i}/{len(unprocessed_ids)}] Fetching Game ID: {game_id}")
    df = fetch_boxscore(game_id)

    if df is not None:
        all_data.append(df)

    # Save every 25 or at the end
    if i % 25 == 0 or i == len(unprocessed_ids):
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(OUTPUT_FILE, index=False)
        print(f"💾 Progress saved: {OUTPUT_FILE} ({len(combined)} rows total)")

    # Delay between games
    time.sleep(REQUEST_DELAY)

    if i % COOLDOWN_EVERY == 0:
        print(f"⏸️ Cooling off for {COOLDOWN_TIME} seconds after {i} games...")
        time.sleep(COOLDOWN_TIME)
