library(hoopR)
library(dplyr)
library(readr)

# Define season string and output path
season <- "2024-25"
output_path <- "C:/Projects/NBA_Prediction/data/raw/core/nba_gamefinder.csv"

# Get all league games for season
gamefinder <- nba_leaguegamefinder(
  season = season,
  league_id_nullable = "00",      # NBA
  season_type_nullable = "Regular Season"
)

# Extract the main data frame
games <- gamefinder$LeagueGameFinderResults

# Optional: Keep only relevant columns
games_cleaned <- games %>%
  select(GAME_ID, GAME_DATE, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME, MATCHUP, WL, MIN, PTS, FG_PCT, REB, AST, TOV)

# Save to CSV
write_csv(games_cleaned, output_path)

cat("✅ Saved game finder data to:", output_path, "\n")
