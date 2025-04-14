# GETALL.R
# Purpose: One script to update all NBA datasets daily, including backfilling missing games

library(fs)
library(glue)
library(hoopR)
library(dplyr)
library(readr)

cat("\n🚀 Starting daily NBA data update...\n")

# --- 1. Get updated list of all games via gamefinder ---
cat("\n🎮 Fetching latest game list from nba_leaguegamefinder()...\n")

season <- "2024-25"
games_path <- "C:/Projects/NBA_Prediction/data/raw/core/games.csv"

gamefinder <- nba_leaguegamefinder(
  season = season,
  league_id_nullable = "00",
  season_type_nullable = "Regular Season"
)

games <- gamefinder$LeagueGameFinderResults %>%
  select(GAME_ID, GAME_DATE, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME, MATCHUP, WL)

games_unique <- games %>%
  distinct(GAME_ID, .keep_all = TRUE)

write_csv(games_unique, games_path)
cat("✅ Saved unique GAME_IDs to:", games_path, "\n")

# --- 2. Update each core dataset ---
cat("\n📦 Fetching core data...\n")
source("scripts/fetch/core_data.R")

# --- 3. Update advanced boxscores ---
cat("\n📈 Fetching advanced box score stats...\n")
source("scripts/fetch/advanced_boxscores.R")

# --- 4. Update hustle and defensive stats ---
cat("\n🛡️ Fetching hustle and defensive data...\n")
source("scripts/fetch/hustle_defense.R")

# --- 5. Update matchups ---
cat("\n🤝 Fetching matchup data...\n")
source("scripts/fetch/matchups.R")

# --- 6. Update tracking and player logs ---
cat("\n🎯 Fetching tracking and player game data...\n")
source("scripts/fetch/tracking_stats.R")

cat("\n✅ All NBA data updated successfully!\n")
