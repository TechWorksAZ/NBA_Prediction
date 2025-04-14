import os
import time
import pandas as pd
import requests
from nba_api.stats.endpoints import ShotChartDetail, LeagueGameFinder

# === CONFIG ===
SEASON = "2024-25"
RAW_DIR = os.path.join("data", "raw", "shot_charts")
os.makedirs(RAW_DIR, exist_ok=True)
RETRIES = 3
SLEEP = 1.0
COOLDOWN_EVERY = 25
COOLDOWN_TIME = 10

# === STEP 1: Build team lookup by GAME_ID ===
print("🔍 Building game-to-team lookup...")
gamefinder = LeagueGameFinder(season_nullable=SEASON, season_type_nullable="Regular Season")
games_df = gamefinder.get_data_frames()[0]
games_df["GAME_ID"] = games_df["GAME_ID"].astype(str)

game_team_map = games_df.groupby("GAME_ID")["TEAM_ID"].apply(list).to_dict()

# === STEP 2: Identify which files are empty or too small ===
print("📂 Scanning existing shot chart files...")
files = os.listdir(RAW_DIR)
retry_ids = []

for file in files:
    if file.endswith(".csv"):
        path = os.path.join(RAW_DIR, file)
        try:
            df = pd.read_csv(path)
            if df.empty or len(df) < 10:
                retry_ids.append(file.replace(".csv", ""))
        except Exception:
            retry_ids.append(file.replace(".csv", ""))

print(f"🔁 {len(retry_ids)} games will be retried due to empty or small shot chart data.")

# === FETCH FUNCTION ===
def fetch_shot_chart(game_id, team_id, retries=RETRIES):
    for attempt in range(retries):
        try:
            chart = ShotChartDetail(
                team_id=team_id,
                game_id_nullable=game_id,
                player_id=0,
                season_type_all_star="Regular Season",
                context_measure_simple="FGA"
            )
            df = chart.get_data_frames()[0]
            if df.empty:
                print(f"⚠️ No data for {game_id} (team {team_id})")
                return None
            df["TEAM_ID"] = team_id
            return df
        except requests.exceptions.ReadTimeout:
            print(f"⏳ Timeout {game_id} (team {team_id}) attempt {attempt + 1}")
            time.sleep(2 + attempt)
        except Exception as e:
            print(f"❌ Error fetching {game_id} (team {team_id}): {e}")
            return None
    return None

# === STEP 3: Retry fetching for each bad file ===
for i, game_id in enumerate(retry_ids, 1):
    team_ids = game_team_map.get(game_id, [])
    if not team_ids:
        print(f"🚫 No team info found for game {game_id}")
        continue

    print(f"[{i}/{len(retry_ids)}] Retrying Game ID: {game_id} (Teams: {team_ids})")
    all_data = []

    for team_id in team_ids:
        df = fetch_shot_chart(game_id, team_id)
        if df is not None:
            all_data.append(df)
        time.sleep(SLEEP)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        out_path = os.path.join(RAW_DIR, f"{game_id}.csv")
        combined.to_csv(out_path, index=False)
        print(f"✅ Re-saved shot chart for {game_id}")
    else:
        print(f"⚠️ Still no data for {game_id}")

    if i % COOLDOWN_EVERY == 0:
        print(f"⏸️ Cooling for {COOLDOWN_TIME} seconds...")
        time.sleep(COOLDOWN_TIME)

print("\n🎉 Retry process complete!")
