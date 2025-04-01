import os
import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder

# === CONFIG ===
SEASON = "2024-25"
SHOT_DIR = os.path.join("data", "raw", "shot_charts")

# === FETCH ALL GAME IDS FROM NBA API ===
print("📥 Fetching official list of NBA games...")
gamefinder = LeagueGameFinder(season_nullable=SEASON, season_type_nullable="Regular Season")
games_df = gamefinder.get_data_frames()[0]
games_df["GAME_ID"] = games_df["GAME_ID"].astype(str)

# === GET GAME_IDS THAT HAVE A SHOT CHART FILE ===
existing_files = {f.replace(".csv", "") for f in os.listdir(SHOT_DIR) if f.endswith(".csv")}
unique_games = games_df["GAME_ID"].unique()
missing_games = [gid for gid in unique_games if gid not in existing_files]

print(f"\n📊 Total games in NBA API: {len(unique_games)}")
print(f"✅ Games with shot charts: {len(existing_files)}")
print(f"❌ Games missing shot charts: {len(missing_games)}")

# === MERGE DETAILS FOR MISSING GAMES ===
missing_df = games_df[games_df["GAME_ID"].isin(missing_games)]
missing_df = missing_df[["GAME_ID", "GAME_DATE", "TEAM_ID", "MATCHUP"]]
missing_df = missing_df.sort_values(["GAME_DATE", "GAME_ID"])

# === DISPLAY & SAVE OUTPUT ===
print("\n🧾 Sample of missing games:")
print(missing_df.head(15))

out_path = os.path.join(SHOT_DIR, "_missing_shot_chart_games.csv")
missing_df.to_csv(out_path, index=False)
print(f"\n💾 Saved missing game report to: {out_path}")
