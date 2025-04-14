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
COOLDOWN_EVERY = 50
COOLDOWN_TIME = 10

# === LOAD EXISTING FILES ===
existing_files = {f.replace(".csv", "") for f in os.listdir(RAW_DIR) if f.endswith(".csv")}
print(f"✅ {len(existing_files)} shot chart files already exist.")

# === GET GAME IDS & TEAM IDS ===
print("📥 Fetching list of NBA games...")
gamefinder = LeagueGameFinder(season_nullable=SEASON, season_type_nullable="Regular Season")
games_df = gamefinder.get_data_frames()[0]
games_df = games_df[(games_df["TEAM_ID"] >= 1610612737) & (games_df["TEAM_ID"] <= 1610612766)]
games_df["GAME_ID"] = games_df["GAME_ID"].astype(str)

# Get unique (game_id, team_id) pairs, grouped by game
grouped = games_df.groupby("GAME_ID")["TEAM_ID"].apply(list).reset_index()
grouped = grouped[~grouped["GAME_ID"].isin(existing_files)]
print(f"🆕 {len(grouped)} games to fetch.")

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
                print(f"⚠️ Empty shot chart for {game_id} (team {team_id})")
                return None
            return df
        except requests.exceptions.ReadTimeout:
            print(f"⏳ Timeout for {game_id} (team {team_id}), retrying ({attempt + 1}/{retries})...")
            time.sleep(2 + attempt)
        except Exception as e:
            print(f"❌ Failed to fetch shot chart for {game_id} (team {team_id}): {e}")
            return None
    return None

# === MAIN LOOP ===
for i, row in enumerate(grouped.itertuples(index=False), 1):
    game_id = row.GAME_ID
    team_ids = row.TEAM_ID

    print(f"[{i}/{len(grouped)}] Fetching shot chart for Game ID: {game_id} ({team_ids})")

    all_team_data = []
    for team_id in team_ids:
        df = fetch_shot_chart(game_id, team_id)
        if df is not None:
            df["TEAM_ID"] = team_id
            all_team_data.append(df)
        time.sleep(SLEEP)

    if all_team_data:
        combined = pd.concat(all_team_data, ignore_index=True)
        out_path = os.path.join(RAW_DIR, f"{game_id}.csv")
        combined.to_csv(out_path, index=False)
        print(f"💾 Saved combined shot chart: {out_path}")
    else:
        print(f"⚠️ Skipped saving — no data for Game ID: {game_id}")

    if i % COOLDOWN_EVERY == 0:
        print(f"⏸️ Cooling off for {COOLDOWN_TIME} seconds...")
        time.sleep(COOLDOWN_TIME)
