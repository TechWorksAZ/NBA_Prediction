# player_ids.R
# Purpose: Fetch current-season NBA player metadata and IDs for 2024-25 only

library(hoopR)
library(readr)
library(glue)
library(fs)

# --- 1. Set output path
save_path <- "C:/Projects/NBA_Prediction/data/raw/core/player_ids_2025.csv"
dir_create(path_dir(save_path))

# --- 2. Fetch player index (current season only, no historical players)
cat("📥 Fetching NBA player index for 2024-25 season only...\n")
tryCatch({
  df <- nba_playerindex(season = year_to_season(2024), historical = 0)[[1]]
  write_csv(df, save_path)
  cat(glue("💾 Saved player index to: {save_path}\n"))
}, error = function(e) {
  cat(glue("❌ Failed to fetch player index: {e$message}\n"))
})
