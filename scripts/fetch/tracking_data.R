# tracking_stats.R
# Purpose: Fetch NBA tracking and dashboard data for 2024-25 season
# Source: hoopR package

library(hoopR)
library(dplyr)
library(readr)
library(glue)
library(fs)
library(purrr)

# --- 1. Set constants
season <- year_to_season(2024)
base_dir <- "C:/Projects/NBA_Prediction/data/raw/tracking"
dir_create(base_dir)
max_retries <- 3
retry_delay <- 2  # seconds

# --- 2. Load game IDs and player IDs
game_ids <- read_csv(
  "C:/Projects/NBA_Prediction/data/raw/core/games.csv"
)$GAME_ID
player_ids <- read_csv(
  "C:/Projects/NBA_Prediction/data/raw/core/nba_player_ids.csv"
)$PERSON_ID

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

# --- 4. Game-level fetch by game_id with retry logic
fetch_game_stat <- function(game_id, fn_name, fetch_fn) {
  save_dir <- file.path(base_dir, fn_name)
  dir_create(save_dir)

  file_path <- file.path(save_dir, glue("{game_id}.csv"))
  if (file_exists(file_path)) {
    # Check if file is empty
    if (file_size(file_path) == 0) {
      file_delete(file_path)  # Remove empty file to retry
    } else {
      cat(glue("✅ {fn_name} already exists for game {game_id}, skipping...\n"))
      return()
    }
  }

  # Retry logic
  for (attempt in 1:max_retries) {
    tryCatch({
      # Get the API response
      response <- fetch_fn(game_id)
      
      # Check if response is valid
      if (is.null(response) || length(response) == 0) {
        cat(glue("⚠️ No data returned for game {game_id} (attempt {attempt}/{max_retries})\n"))
        if (attempt < max_retries) {
          Sys.sleep(retry_delay)
          next
        }
        return()
      }
      
      # Extract the data frame
      df <- response[[1]]
      
      # Check if data frame is valid
      if (is.null(df) || nrow(df) == 0) {
        cat(glue("⚠️ Empty data frame for game {game_id} (attempt {attempt}/{max_retries})\n"))
        if (attempt < max_retries) {
          Sys.sleep(retry_delay)
          next
        }
        return()
      }
      
      # Only save if we have valid data
      write_csv(df, file_path)
      cat(glue("💾 Saved {fn_name} for game {game_id} with {nrow(df)} rows\n"))
      return()  # Success, exit the retry loop
      
    }, error = function(e) {
      cat(glue("❌ Failed {fn_name} for game {game_id} (attempt {attempt}/{max_retries}): {e$message}\n"))
      if (attempt < max_retries) {
        Sys.sleep(retry_delay)
      }
    })
  }
}

# --- 5. Player-level fetch by player_id
fetch_player_stat <- function(player_id, fn_name, fetch_fn) {
  save_dir <- file.path(base_dir, fn_name)
  dir_create(save_dir)

  file_path <- file.path(save_dir, glue("{player_id}.csv"))
  if (file_exists(file_path)) {
    cat(glue("✅ {fn_name} already exists for player {player_id}, skipping..\n"))
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
fetch_season_data(
  "leaguedashteamstats",
  function() nba_leaguedashteamstats(season = season)
)
fetch_season_data(
  "leaguedashplayerstats",
  function() nba_leaguedashplayerstats(season = season)
)
fetch_season_data(
  "leaguedashptstats",
  function() nba_leaguedashptstats(season = season)
)
fetch_season_data(
  "leaguedashptdefend",
  function() nba_leaguedashptdefend(season = season)
)
fetch_season_data(
  "leaguedashteamclutch",
  function() nba_leaguedashteamclutch(season = season)
)
fetch_season_data(
  "teamshootingsplits",
  function() nba_teamdashboardbyshootingsplits(season = season)
)

# --- 7. Fetch player gamelogs by player_id
cat("\n🎯 Fetching player game logs...\n")
for (player_id in player_ids) {
  fetch_player_stat(
    player_id,
    "playergamelogs",
    function(id) nba_playergamelogs(player_id = id, season = season)
  )
}

# --- 8. Fetch shot chart details by game_id
cat("\n🎯 Fetching shot chart data by game...\n")

# Create output directories if they don't exist
details_dir <- file.path(base_dir, "shotchartdetail", "details")
averages_dir <- file.path(base_dir, "shotchartdetail", "league_averages")
dir_create(details_dir)
dir_create(averages_dir)

for (game_id in game_ids) {
  # Format game ID to match NBA API requirements (e.g., "0022401133")
  formatted_game_id <- sprintf("00%08d", as.numeric(game_id))
  
  # Check if files already exist
  details_file <- file.path(details_dir, glue("shot_chart_{formatted_game_id}.csv"))
  averages_file <- file.path(averages_dir, glue("league_averages_{formatted_game_id}.csv"))
  
  if (file_exists(details_file) && file_exists(averages_file)) {
    cat(glue("✅ Shot chart files already exist for game {formatted_game_id}, skipping...\n"))
    next
  }
  
  # Fetch shot chart data
  tryCatch({
    shot_data <- nba_shotchartdetail(
      game_id = formatted_game_id,
      player_id = 0,  # 0 for all players
      season = season,
      league_id = '00',  # NBA
      season_type = season_type
    )
    
    # Save the data to CSV files if available
    if (!is.null(shot_data$Shot_Chart_Detail)) {
      write_csv(shot_data$Shot_Chart_Detail, details_file)
      cat(glue("💾 Saved shot chart details for game {formatted_game_id}\n"))
      
      if (!is.null(shot_data$LeagueAverages)) {
        write_csv(shot_data$LeagueAverages, averages_file)
        cat(glue("💾 Saved league averages for game {formatted_game_id}\n"))
      }
    } else {
      cat(glue("⚠️ No shot chart details available for game {formatted_game_id}\n"))
    }
    
    # Add a small delay to avoid rate limiting
    Sys.sleep(1)
    
  }, error = function(e) {
    cat(glue("❌ Failed to fetch shot chart for game {formatted_game_id}: {e$message}\n"))
  })
}

cat("\n✅ Tracking and dashboard data collection complete!\n")
