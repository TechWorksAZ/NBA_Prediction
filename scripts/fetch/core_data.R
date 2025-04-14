# core_data.R
# Purpose: Fetch and save core NBA game data from hoopR + NBA API
# Source: hoopR package & NBA Stats API

library(hoopR)
library(dplyr)
library(readr)
library(lubridate)
library(glue)
library(fs)

# Define your raw data directory
raw_data_dir <- "C:/projects/nba_prediction/data/raw/core"
dir_create(raw_data_dir)

# -------- 1. Load NBA Schedule (hoopR) --------
cat("\n📅 Fetching latest NBA schedule...\n")
tryCatch({
  schedule <- load_nba_schedule()
  write_csv(schedule, file.path(raw_data_dir, "nba_schedule.csv"))
  cat("✅ Schedule saved\n")
}, error = function(e) cat("❌ Failed to fetch schedule:", e$message, "\n"))

# -------- 2. Load NBA Team Box Scores --------
cat("\n📊 Fetching latest team box scores...\n")
tryCatch({
  team_box <- load_nba_team_box()
  write_csv(team_box, file.path(raw_data_dir, "nba_team_box_scores.csv"))
  cat("✅ Team box scores saved\n")
}, error = function(e) cat("❌ Failed to fetch team box scores:", e$message, "\n"))

# -------- 3. Load NBA Player Box Scores --------
cat("\n👤 Fetching latest player box scores...\n")
tryCatch({
  player_box <- load_nba_player_box()
  write_csv(player_box, file.path(raw_data_dir, "nba_player_box_scores.csv"))
  cat("✅ Player box scores saved\n")
}, error = function(e) cat("❌ Failed to fetch player box scores:", e$message, "\n"))

# -------- 4. Load NBA Play-by-Play Data --------
cat("\n🎬 Fetching latest play-by-play data...\n")
tryCatch({
  pbp <- load_nba_pbp()
  write_csv(pbp, file.path(raw_data_dir, "nba_play_by_play.csv"))
  cat("✅ Play-by-play data saved\n")
}, error = function(e) cat("❌ Failed to fetch play-by-play data:", e$message, "\n"))

# -------- 5. Load NBA Team Game Logs (NBA API) --------
cat("\n📘 Fetching latest team game logs...\n")
tryCatch({
  logs <- nba_teamgamelog(season = 2024)[[1]]
  write_csv(logs, file.path(raw_data_dir, "nba_team_game_logs.csv"))
  cat("✅ Team game logs saved\n")
}, error = function(e) cat("❌ Failed to fetch team game logs:", e$message, "\n"))

cat("\n🎉 Core data fetch completed.\n")
