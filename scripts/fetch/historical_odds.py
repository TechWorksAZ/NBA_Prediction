# scripts/fetch/nba_scrape_odds.py

from scraper.sbr_historical_odds_package.odds_scraper import NBAOddsScraper
import pandas as pd
import os

# Setup
season_years = [2025]
output_dir = "data/raw/betting"
os.makedirs(output_dir, exist_ok=True)
merged_file = os.path.join(output_dir, "nba_betting_odds_2025.csv")

# Scrape
print(f"📦 Scraping NBA odds for seasons: {season_years}")
scraper = NBAOddsScraper(season_years)
df_new = scraper.driver()
df_new["date"] = pd.to_datetime(df_new["date"], format="%Y%m%d")

# Merge with existing if available
if os.path.exists(merged_file):
    df_existing = pd.read_csv(merged_file, parse_dates=["date"])
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["season", "date", "home_team", "away_team"])
    print(f"🔄 Merged with existing file. Final shape: {df_all.shape}")
else:
    df_all = df_new
    print(f"📄 No existing file found. Saving fresh scrape. Rows: {df_new.shape[0]}")

# Save merged file
df_all.to_csv(merged_file, index=False)
print(f"✅ Saved merged NBA odds to: {merged_file}")
