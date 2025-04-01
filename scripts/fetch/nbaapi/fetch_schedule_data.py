import os
import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder

# === PATHS ===
RAW_DATA_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, "game_schedule_2025.csv")

# === LOAD EXISTING FILE (IF ANY) ===
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE, dtype={"GAME_ID": str})
    existing_ids = set(existing_df["GAME_ID"].astype(str).unique())
    print(f"✅ Loaded {len(existing_df)} rows from existing schedule.")
else:
    existing_df = pd.DataFrame()
    existing_ids = set()

# === FETCH FRESH DATA ===
print("📅 Fetching NBA game schedule (2024–25)...")
finder = LeagueGameFinder(season_nullable="2024-25", season_type_nullable="Regular Season")
games_df = finder.get_data_frames()[0]
games_df = games_df[(games_df["TEAM_ID"] >= 1610612737) & (games_df["TEAM_ID"] <= 1610612766)]

# Keep only needed columns
games_df["GAME_ID"] = games_df["GAME_ID"].astype(str)
games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])
games_df["IS_HOME"] = games_df["MATCHUP"].str.contains("vs.")
games_df["TEAM_ABBREVIATION"] = games_df["TEAM_ABBREVIATION"].astype(str)

# Get opponent team ID
def extract_opponent(row):
    opponent_row = games_df[
        (games_df["GAME_ID"] == row["GAME_ID"]) &
        (games_df["TEAM_ID"] != row["TEAM_ID"])
    ]
    return opponent_row["TEAM_ID"].values[0] if not opponent_row.empty else None

games_df["OPPONENT_TEAM_ID"] = games_df.apply(extract_opponent, axis=1)

# === Compute REST DAYS & BACK-TO-BACK ===
print("🧠 Calculating rest days and back-to-backs...")
games_df.sort_values(by=["TEAM_ID", "GAME_DATE"], inplace=True)
games_df["REST_DAYS"] = games_df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.fillna(-1).astype(int)
games_df["BACK_TO_BACK"] = games_df["REST_DAYS"] == 0

# === FILTER NEW GAMES ONLY ===
new_games = games_df[~games_df["GAME_ID"].isin(existing_ids)]
print(f"🆕 New games found: {len(new_games)}")

# === COMBINE & SAVE ===
combined = pd.concat([existing_df, new_games], ignore_index=True)
combined.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], inplace=True)
combined.to_csv(OUTPUT_FILE, index=False)
print(f"💾 Saved updated schedule to {OUTPUT_FILE} ({len(combined)} total rows).")
