import os
import time
import pandas as pd
import requests
from nba_api.stats.endpoints import (
    LeagueGameFinder,
    BoxScoreAdvancedV2,
    BoxScoreFourFactorsV2,
    BoxScoreMiscV2
)

# === CONFIG ===
REQUEST_DELAY = 0.75
COOLDOWN_EVERY = 50
COOLDOWN_TIME = 10
RETRIES = 3

# === PATH SETUP ===
RAW_DATA_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

ADV_FILE = os.path.join(RAW_DATA_DIR, "team_box_scores_advanced_2025.csv")
FOUR_FILE = os.path.join(RAW_DATA_DIR, "team_box_scores_fourfactors_2025.csv")
MISC_FILE = os.path.join(RAW_DATA_DIR, "team_box_scores_misc_2025.csv")

# === LOAD EXISTING DATA ===
def load_existing(filepath):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, dtype={"GAME_ID": str})
        print(f"✅ Loaded {len(df)} rows from {os.path.basename(filepath)}")
        return df, set(df['GAME_ID'].astype(str).unique())
    else:
        return pd.DataFrame(), set()

adv_df, adv_ids = load_existing(ADV_FILE)
four_df, four_ids = load_existing(FOUR_FILE)
misc_df, misc_ids = load_existing(MISC_FILE)

# === GET GAME IDS ===
print("📥 Fetching list of NBA games...")
gamefinder = LeagueGameFinder(season_nullable="2024-25", season_type_nullable="Regular Season")
games_df = gamefinder.get_data_frames()[0]
games_df = games_df[(games_df["TEAM_ID"] >= 1610612737) & (games_df["TEAM_ID"] <= 1610612766)]
all_game_ids = [str(gid) for gid in games_df["GAME_ID"].unique()]
unprocessed_ids = [
    gid for gid in all_game_ids
    if gid not in adv_ids or gid not in four_ids or gid not in misc_ids
]

print(f"🔄 Advanced: {len(adv_ids)} | FourFactors: {len(four_ids)} | Misc: {len(misc_ids)}")
print(f"🆕 Total games to fetch: {len(unprocessed_ids)}")

# === FETCH FUNCTION ===
def fetch_boxscore(endpoint_cls, game_id, table_index=1, retries=RETRIES):
    for attempt in range(1, retries + 1):
        try:
            result = endpoint_cls(game_id=game_id)
            df = result.get_data_frames()[table_index]
            if df.empty:
                print(f"⚠️ Empty response from {endpoint_cls.__name__} for {game_id}")
                return None
            return df
        except Exception as e:
            print(f"⛔ Error fetching {game_id} from {endpoint_cls.__name__}: {e}")
            if attempt < retries:
                backoff = 2 * attempt
                print(f"🔁 Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                return None

# === MAIN LOOP ===
for i, game_id in enumerate(unprocessed_ids, 1):
    print(f"\n[{i}/{len(unprocessed_ids)}] Game ID: {game_id}")

    updated = False

    if game_id not in adv_ids:
        adv_data = fetch_boxscore(BoxScoreAdvancedV2, game_id)
        if adv_data is not None and not adv_data.empty:
            adv_df = pd.concat([adv_df, adv_data], ignore_index=True)
            adv_df.to_csv(ADV_FILE, index=False)
            print(f"💾 Saved Advanced for {game_id}")
            updated = True

    if game_id not in four_ids:
        four_data = fetch_boxscore(BoxScoreFourFactorsV2, game_id)
        if four_data is not None and not four_data.empty:
            four_df = pd.concat([four_df, four_data], ignore_index=True)
            four_df.to_csv(FOUR_FILE, index=False)
            print(f"💾 Saved Four Factors for {game_id}")
            updated = True

    if game_id not in misc_ids:
        misc_data = fetch_boxscore(BoxScoreMiscV2, game_id)
        if misc_data is not None and not misc_data.empty:
            misc_df = pd.concat([misc_df, misc_data], ignore_index=True)
            misc_df.to_csv(MISC_FILE, index=False)
            print(f"💾 Saved Misc for {game_id}")
            updated = True

    if updated:
        time.sleep(REQUEST_DELAY)

    if i % COOLDOWN_EVERY == 0:
        print(f"⏸️ Cooling off for {COOLDOWN_TIME} seconds...")
        time.sleep(COOLDOWN_TIME)
