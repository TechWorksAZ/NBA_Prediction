# advanced_boxscores.R
# Purpose: Fetch and save advanced NBA box score data from hoopR
# Source: hoopR package

library(hoopR)
library(dplyr)
library(readr)
library(lubridate)
library(purrr)
library(glue)
library(janitor)
library(tidyr)
library(fs)

# Define directories
raw_data_dir <- "C:/projects/nba_prediction/data/raw/advanced"
dir_create(raw_data_dir, recurse = TRUE)
game_ids_file <- "C:/projects/nba_prediction/data/raw/core/games.csv"

# Create logs directory if it doesn't exist
dir_create("C:/projects/nba_prediction/logs", recurse = TRUE)

# Output + Log setup for all 7 boxscore types
boxscore_types <- list(
  advanced = list(
    prefix = "boxscoreadvancedv3",
    folders = c("home_player", "away_player", "home_totals", "away_totals"),
    log = "advanced_boxscoreadavanced.log",
    fields = c(
      home_team_player_advanced = "home_player",
      away_team_player_advanced = "away_player",
      home_team_totals_advanced = "home_totals",
      away_team_totals_advanced = "away_totals"
    )
  ),
  fourfactors = list(
    prefix = "boxscorefourfactorsv3",
    folders = c("home_player", "away_player", "home_totals", "away_totals"),
    log = "advanced_boxscorefourfactors.log",
    fields = c(
      home_team_player_four_factors = "home_player",
      away_team_player_four_factors = "away_player",
      home_team_totals_four_factors = "home_totals",
      away_team_totals_four_factors = "away_totals"
    )
  ),
  misc = list(
    prefix = "boxscoremiscv3",
    folders = c("home_player", "away_player", "home_totals", "away_totals"),
    log = "advanced_boxscoremisc.log",
    fields = c(
      home_team_player_misc = "home_player",
      away_team_player_misc = "away_player",
      home_team_totals_misc = "home_totals",
      away_team_totals_misc = "away_totals"
    )
  ),
  playertrack = list(
    prefix = "boxscoreplayertrackv3",
    folders = c("home_player", "away_player", "home_totals", "away_totals"),
    log = "advanced_boxscoreplayertrack.log",
    fields = c(
      home_team_player_player_track = "home_player",
      away_team_player_player_track = "away_player",
      home_team_totals_player_track = "home_totals",
      away_team_totals_player_track = "away_totals"
    )
  ),
  scoring = list(
    prefix = "boxscorescoringv3",
    folders = c("home_player", "away_player", "home_totals", "away_totals"),
    log = "advanced_boxscorescoring.log",
    fields = c(
      home_team_player_scoring = "home_player",
      away_team_player_scoring = "away_player",
      home_team_totals_scoring = "home_totals",
      away_team_totals_scoring = "away_totals"
    )
  ),
  traditional = list(
    prefix = "boxscoretraditionalv3",
    folders = c(
      "home_player", "away_player",
      "home_totals", "away_totals",
      "home_starter_totals", "away_starter_totals",
      "home_bench_totals", "away_bench_totals"
    ),
    log = "advanced_boxscoretraditional.log",
    fields = c(
      home_team_player_traditional = "home_player",
      away_team_player_traditional = "away_player",
      home_team_totals_traditional = "home_totals",
      away_team_totals_traditional = "away_totals",
      home_team_starters_totals = "home_starter_totals",
      away_team_starters_totals = "away_starter_totals",
      home_team_bench_totals = "home_bench_totals",
      away_team_bench_totals = "away_bench_totals"
    )
  ),
  usage = list(
    prefix = "boxscoreusagev3",
    folders = c("home_player", "away_player", "home_totals", "away_totals"),
    log = "advanced_boxscoreusage.log",
    fields = c(
      home_team_player_usage = "home_player",
      away_team_player_usage = "away_player",
      home_team_totals_usage = "home_totals",
      away_team_totals_usage = "away_totals"
    )
  )
)

# Create output dirs
for (bs in boxscore_types) {
  for (folder in bs$folders) {
    dir.create(file.path(raw_data_dir, bs$prefix, folder), recursive = TRUE, showWarnings = FALSE)
  }
}

# Load game IDs
games_df <- read_csv(game_ids_file, show_col_types = FALSE)
game_ids <- unique(games_df$GAME_ID)

# Filter out game IDs that start with 001
game_ids <- game_ids[!grepl("^001", game_ids)]
cat(glue("\n📊 Processing {length(game_ids)} games (excluding preseason games)\n"))

# Loop through each game ID
for (gid in game_ids) {
  cat(glue("\n\n🔄 Processing game {gid}..."))
  
  # Loop through each boxscore type
  for (type in names(boxscore_types)) {
    bs <- boxscore_types[[type]]
    cat(glue("\n📦 Processing {bs$prefix}..."))
    
    # Initialize paths for this game
    paths <- list()
    for (field in names(bs$fields)) {
      fname <- paste0(gid, ".csv")
      folder <- bs$fields[[field]]
      path <- file.path(raw_data_dir, bs$prefix, folder, fname)
      paths[[field]] <- as.character(path)
    }
    
    # Check which files already exist
    existing_files <- sapply(paths, file.exists)
    if (all(existing_files)) {
      cat(glue("\n⏩ {bs$prefix} - {gid} already complete"))
      next
    }
    
    tryCatch({
      func_name <- paste0("nba_", bs$prefix)
      df_list <- do.call(func_name, list(game_id = gid))
      
      # Check if we got valid data
      if (is.null(df_list) || length(df_list) == 0) {
        stop("No data returned from API")
      }
      
      for (field in names(paths)) {
        # Only process files that don't exist
        if (!existing_files[[field]] && !is.null(df_list[[field]])) {
          write_csv(df_list[[field]], paths[[field]])
          cat(glue("\n✅ Saved {field} for {gid}"))
        } else if (existing_files[[field]]) {
          cat(glue("\n⏩ Skipping {field} for {gid} (already exists)"))
        }
      }
    }, error = function(e) {
      # Log the error but don't try to use df_list
      msg <- glue("{Sys.time()} | Failed for {gid} in {bs$prefix} | {e$message}\n")
      write(msg, file = file.path("C:/projects/nba_prediction/logs", bs$log), append = TRUE)
      cat(glue("\n❌ Error: {e$message}"))
      
      # If we have existing files, don't treat this as a complete failure
      if (any(existing_files)) {
        cat(glue("\n⚠️ Some files exist for {gid} in {bs$prefix}, skipping..."))
      }
    })
  }
}

cat("\n🎉 All advanced boxscore types fetched and saved.\n")