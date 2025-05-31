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
cat("\n📅 Fetching NBA schedule...\n")

# Get year from command line argument or use current year
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  year <- as.numeric(args[1])
  if (is.na(year) || year < 2000 || year > 2100) {
    stop("Invalid year provided. Please provide a valid year (e.g., 2024, 2025)")
  }
} else {
  year <- 2024
}

# Convert year to season format (e.g., 2024 -> 2024-25)
season <- year_to_season(year)

tryCatch({
  # Fetch schedule for specified season
  schedule <- nba_schedule(
    league_id = '00',  # NBA games only
    season = season
  )
  
  # Filter out preseason games (season_type_id = 1)
  schedule <- schedule %>%
    filter(season_type_id != 1)
  
  # Save the schedule
  write_csv(schedule, file.path(raw_data_dir, "nba_schedule.csv"))
  cat(glue("✅ Schedule saved for season {season}\n"))
  cat(glue("📊 Total games: {nrow(schedule)}\n"))
  cat(glue("📅 Date range: {min(schedule$game_date)} to {max(schedule$game_date)}\n"))
  
}, error = function(e) {
  cat(glue("❌ Failed to fetch schedule: {e$message}\n"))
})

