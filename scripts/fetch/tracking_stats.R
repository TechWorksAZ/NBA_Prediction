# tracking_stats.R
# Purpose: Fetch NBA tracking and dashboard data for 2024-25 season and save to /data/raw/tracking/

library(hoopR)
library(dplyr)
library(readr)
library(glue)
library(fs)
library(purrr)

# --- 1. Set constants
season <- year_to_season(2024)
season_type <- "Regular Season"
base_dir <- "C:/Projects/NBA_Prediction/data/raw/tracking"
dir_create(base_dir)

# --- 2. Load game IDs and player IDs
game_ids <- read_csv("C:/Projects/NBA_Prediction/data/raw/core/games.csv")$game_id
player_ids <- read_csv("C:/Projects/NBA_Prediction/data/raw/core/player_ids_2025.csv")$PERSON_ID

# --- 3. Helper to fetch and save full-season data
fetch_season_data <- function(fn_name, fetch_fn) {
  save_path <- file.path(base_dir, glue("{fn_name}_{season}.csv"))
  if (file_exists(save_path)) {
    cat(glue("✅ {fn_name} already exists for {season}, skipping...\n"))
    return()
  }

  tryCatch({
    df <- fetch_fn()[[1]]
    write_csv(df, save_path)
    cat(glue("💾 Saved {fn_name} for {season}\n"))
  }, error = function(e) {
    cat(glue("❌ Failed {fn_name} for {season}: {e$message}\n"))
  })
}

# --- 4. Game-level fetch by game_id
fetch_game_stat <- function(game_id, fn_name, fetch_fn) {
  save_dir <- file.path(base_dir, fn_name)
  dir_create(save_dir)

  file_path <- file.path(save_dir, glue("{game_id}.csv"))
  if (file_exists(file_path)) {
    cat(glue("✅ {fn_name} already exists for game {game_id}, skipping...\n"))
    return()
  }

  tryCatch({
    df <- fetch_fn(game_id)[[1]]
    write_csv(df, file_path)
    cat(glue("💾 Saved {fn_name} for game {game_id}\n"))
  }, error = function(e) {
    cat(glue("❌ Failed {fn_name} for game {game_id}: {e$message}\n"))
  })
}

# --- 5. Player-level fetch by player_id
fetch_player_stat <- function(player_id, fn_name, fetch_fn) {
  save_dir <- file.path(base_dir, fn_name)
  dir_create(save_dir)

  file_path <- file.path(save_dir, glue("{player_id}.csv"))
  if (file_exists(file_path)) {
    cat(glue("✅ {fn_name} already exists for player {player_id}, skipping...\n"))
    return()
  }

  tryCatch({
    df <- fetch_fn(player_id)[[1]]
    write_csv(df, file_path)
    cat(glue("💾 Saved {fn_name} for player {player_id}\n"))
  }, error = function(e) {
    cat(glue("❌ Failed {fn_name} for player {player_id}: {e$message}\n"))
  })
}

# --- 6. Fetch season-wide league dashboards
cat("📊 Fetching league-wide dashboards...\n")
fetch_season_data("leaguedashteamstats", function() nba_leaguedashteamstats(season = season))
fetch_season_data("leaguedashplayerstats", function() nba_leaguedashplayerstats(season = season))
fetch_season_data("leaguedashptstats", function() nba_leaguedashptstats(season = season))
fetch_season_data("leaguedashptdefend", function() nba_leaguedashptdefend(season = season))
fetch_season_data("leaguedashteamclutch", function() nba_leaguedashteamclutch(season = season))
fetch_season_data("teamshootingsplits", function() nba_teamdashboardbyshootingsplits(season = season))

# --- 7. Fetch player gamelogs by player_id
cat("\n🎯 Fetching player game logs...\n")
for (player_id in player_ids) {
  fetch_player_stat(player_id, "playergamelogs", function(id) nba_playergamelogs(player_id = id, season = season))
}

# --- 8. Fetch shot chart details by game_id
cat("\n🎯 Fetching shot chart data by game...\n")
for (game_id in game_ids) {
  fetch_game_stat(game_id, "shotchartdetail", function(id) nba_shotchartdetail(game_id = id, season = season))
}

cat("\n✅ Tracking and dashboard data collection complete!\n")
