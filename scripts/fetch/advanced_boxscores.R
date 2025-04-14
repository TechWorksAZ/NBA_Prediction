# advanced_boxscores.R
# Purpose: Fetch advanced NBA box score data (v3) and save to /data/raw/advanced/

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
base_dir <- "C:/Projects/NBA_Prediction/data/raw/advanced"
dir_create(base_dir)

# --- 3. Define helper function to check + fetch each dataset
fetch_and_save <- function(game_id, fn_name, fetch_fn) {
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

# --- 4. Loop through game IDs
for (game_id in game_ids) {
  cat(glue("\n📦 Fetching data for game: {game_id}\n"))

  fetch_and_save(game_id, "boxscorefourfactorsv3", nba_boxscorefourfactorsv3)
  fetch_and_save(game_id, "boxscoreadvancedv3", nba_boxscoreadvancedv3)
  fetch_and_save(game_id, "boxscoremiscv3", nba_boxscoremiscv3)
  fetch_and_save(game_id, "boxscorescoringv3", nba_boxscorescoringv3)
  fetch_and_save(game_id, "boxscoreplayertrackv3", nba_boxscoreplayertrackv3)
  fetch_and_save(game_id, "boxscoreusagev3", nba_boxscoreusagev3)
  fetch_and_save(game_id, "boxscoretraditionalv3", nba_boxscoretraditionalv3)
}

cat("🎉 Advanced box score data collection complete.\n")
