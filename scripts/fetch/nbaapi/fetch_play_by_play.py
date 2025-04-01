import os
import time
import pandas as pd
import requests
from nba_api.stats.endpoints import PlayByPlayV2, LeagueGameFinder

# === PATHS ===
RAW_DIR = os.path.join("data", "raw", "play_by_play")
os.makedirs(RAW_DIR, exist_ok=True)

# === GET EXISTING GAME FILES ===
existing_files = {f.replace(".csv", "") for f in os.listdir(RAW_DIR) if f.endswith(".csv")}
print(f"✅ {len(existing_files)} play-by-play files already exist.")

# === GET GAME IDS FOR 2024–25 ===
print("📥 Fetching game list for 2024–25...")
gamefinder = LeagueGameFinder(season_nullable="2024-25", season_type_nullable="Regular Season")
games_df = gamefinder.get_data_frames()[0]
games_df = games_df[(games_df["TEAM_ID"] >= 1610612737) & (games_df["TEAM_ID"] <= 1610612766)]
all_game_ids = sorted(games_df["GAME_ID"].astype(str).unique())
unprocessed_ids = [gid for gid in all_game_ids if gid not in existing_files]
print(f"🆕 {len(unprocessed_ids)} new games to fetch.")

# === FETCH FUNCTION ===
def fetch_pbp(game_id, retries=3):
    for attempt in range(retries):
        try:
            data = PlayByPlayV2(game_id=game_id).get_data_frames()[0]
            if data.empty:
                print(f"⚠️ Empty PBP for {game_id}")
                return None
            return data
        except requests.exceptions.ReadTimeout:
            print(f"⏳ Timeout for {game_id} (attempt {attempt+1})")
            time.sleep(2 + attempt)
        except Exception as e:
            print(f"❌ Failed to fetch {game_id}: {e}")
            return None
    return None

# === MAIN LOOP ===
for i, game_id in enumerate(unprocessed_ids, 1):
    print(f"[{i}/{len(unprocessed_ids)}] Fetching PBP for Game ID: {game_id}")
    df = fetch_pbp(game_id)

    if df is not None:
        out_path = os.path.join(RAW_DIR, f"{game_id}.csv")
        df.to_csv(out_path, index=False)
        print(f"💾 Saved to {out_path}")

    time.sleep(1.0)
    if i % 50 == 0:
        print("⏸️ Cooling off for 10 seconds...")
        time.sleep(10)
