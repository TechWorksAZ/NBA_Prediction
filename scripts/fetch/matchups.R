# matchups.R
# Purpose: Fetch NBA matchups data and save to /data/raw/matchups/

library(hoopR)
library(dplyr)
library(readr)
library(glue)
library(fs)
library(purrr)

# --- 1. Load all valid NBA game IDs from cached CSV
game_ids <- read_csv("C:/Projects/NBA_Prediction/data/raw/core/games.csv")$game_id
cat(glue("🔢 {length(game_ids)} total game_ids loaded from CSV.\n"))

# --- 2. Set base output directory
base_dir <- "C:/Projects/NBA_Prediction/data/raw/matchups"
dir_create(base_dir)

# --- 3. Helper to fetch & save per-game data
fetch_and_save_game <- function(game_id, fn_name, fetch_fn) {
  save_dir <- file.path(base_dir, fn_name)
  file_path <- file.path(save_dir, glue("{game_id}.csv"))
  dir_create(save_dir)

  if (file_exists(file_path)) {
    cat(glue("✅ {fn_name} already exists for {game_id}, skipping...\n"))
    return(NULL)
  }

  tryCatch({
    df <- fetch_fn(game_id)[[1]]
    write_csv(df, file_path)
    cat(glue("💾 Saved {fn_name} for {game_id}\n"))
  }, error = function(e) {
    cat(glue("❌ Failed {fn_name} for game {game_id}: {e$message}\n"))
  })
}

# --- 4. Helper to fetch & save season-level data
fetch_and_save_season <- function(fn_name, fetch_fn, season = 2024) {
  save_dir <- file.path(base_dir, fn_name)
  file_path <- file.path(save_dir, glue("{season}.csv"))
  dir_create(save_dir)

  if (file_exists(file_path)) {
    cat(glue("✅ {fn_name} already exists for season {season}, skipping...\n"))
    return(NULL)
  }

  tryCatch({
    df <- fetch_fn(season = season)[[1]]
    write_csv(df, file_path)
    cat(glue("💾 Saved {fn_name} for season {season}\n"))
  }, error = function(e) {
    cat(glue("❌ Failed {fn_name} for season {season}: {e$message}\n"))
  })
}

# --- 5. Fetch game-level matchups
for (game_id in game_ids) {
  cat(glue("\n🎯 Fetching matchups for game: {game_id}\n"))
  fetch_and_save_game(game_id, "boxscorematchupsv3", nba_boxscorematchupsv3)
}

# --- 6. Fetch season-level matchups rollup
cat("\n📊 Fetching matchup rollup for 2024 season...\n")
fetch_and_save_season("matchupsrollup", nba_matchupsrollup, season = year_to_season(2024))

cat("🏁 Matchup data fetch complete.\n")
