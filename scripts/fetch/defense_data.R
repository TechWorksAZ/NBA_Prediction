# hustle_defense.R
# Purpose: Fetch NBA hustle and defensive data and save to /data/raw/defense/

library(hoopR)
library(dplyr)
library(readr)
library(glue)
library(fs)
library(purrr)

# --- 1. Load all valid NBA game IDs from cached CSV
game_ids <- read_csv(
  "C:/Projects/NBA_Prediction/data/raw/core/games.csv",
  show_col_types = FALSE
)$GAME_ID
cat(glue("🔢 {length(game_ids)} total game_ids loaded from CSV.\n"))

# --- 2. Set base output directory
base_dir <- "C:/Projects/NBA_Prediction/data/raw/defense"
dir_create(base_dir)

# --- 3. Define helper for game-by-game data
fetch_and_save_game <- function(game_id, fn_name, fetch_fn) {
  if (fn_name == "hustlestatsboxscore") {
    save_dir <- file.path(base_dir, fn_name)
    player_path <- as.character(file.path(save_dir, "player", glue("{game_id}.csv")))
    team_path <- as.character(file.path(save_dir, "team", glue("{game_id}.csv")))
    dir_create(dirname(player_path))
    dir_create(dirname(team_path))

    if (file.exists(player_path) && file.exists(team_path)) {
      cat(glue("✅ {fn_name} already exists for {game_id}, skipping...\n"))
      return(NULL)
    }

    tryCatch({
      df_list <- fetch_fn(game_id)
      player_df <- df_list[["PlayerStats"]]
      team_df <- df_list[["TeamStats"]]

      if (!is.null(player_df) && nrow(player_df) > 0) {
        write_csv(player_df, player_path)
        cat(glue("💾 Saved PlayerStats for {game_id}\n"))
      } else {
        cat(glue("⚠️ No PlayerStats for {game_id}, skipping save.\n"))
      }

      if (!is.null(team_df) && nrow(team_df) > 0) {
        write_csv(team_df, team_path)
        cat(glue("💾 Saved TeamStats for {game_id}\n"))
      } else {
        cat(glue("⚠️ No TeamStats for {game_id}, skipping save.\n"))
      }
    }, error = function(e) {
      cat(glue("❌ Failed {fn_name} for game {game_id}: {e$message}\n"))
    })

  } else if (fn_name == "boxscoredefensivev2") {
    save_base <- file.path(base_dir, fn_name)
    paths <- list(
      home_team_player_defensive = as.character(file.path(save_base, "home_player", glue("{game_id}.csv"))),
      away_team_player_defensive = as.character(file.path(save_base, "away_player", glue("{game_id}.csv")))
    )

    # Skip only if ALL files already exist
    if (all(file.exists(unlist(paths)))) {
      cat(glue("✅ {fn_name} already exists for {game_id}, skipping...\n"))
      return(NULL)
    }

    tryCatch({
      df_list <- fetch_fn(game_id)
      
      for (field in names(paths)) {
        if (!file.exists(paths[[field]]) && !is.null(df_list[[field]])) {
          dir_create(dirname(paths[[field]]))
          write_csv(df_list[[field]], paths[[field]])
          cat(glue("💾 Saved {field} for {game_id}\n"))
        }
      }
    }, error = function(e) {
      cat(glue("❌ Failed {fn_name} for game {game_id}: {e$message}\n"))
    })

  } else {
    save_dir <- file.path(base_dir, fn_name)
    file_path <- as.character(file.path(save_dir, glue("{game_id}.csv")))
    dir_create(save_dir)

    if (file.exists(file_path)) {
      cat(glue("✅ {fn_name} already exists for {game_id}, skipping...\n"))
      return(NULL)
    }

    tryCatch({
      df_list <- fetch_fn(game_id)
      df <- df_list[[1]]
      if (is.null(df) || nrow(df) == 0 || ncol(df) <= 2) {
        cat(glue("⚠️ {fn_name} returned no useful data for {game_id}, ",
                 "skipping save.\n"))
        return(NULL)
      }
      write_csv(df, file_path)
      cat(glue("💾 Saved {fn_name} for {game_id}\n"))
    }, error = function(e) {
      cat(glue("❌ Failed {fn_name} for game {game_id}: {e$message}\n"))
    })
  }
}

# --- 4. Define helper for league-wide season-level data
fetch_and_save_season <- function(fn_name, fetch_fn) {
  save_dir <- file.path(base_dir, fn_name)
  file_path <- as.character(file.path(save_dir, "2025.csv"))
  dir_create(save_dir)

  if (file.exists(file_path)) {
    cat(glue("✅ {fn_name} already exists for season 2025, skipping...\n"))
    return(NULL)
  }

  tryCatch({
    df <- fetch_fn(date_from = "2024-10-22", date_to = "2025-04-06")[[1]]
    cat(glue("📐 {fn_name} for 2025: {nrow(df)} rows x {ncol(df)} columns\n"))
    if (nrow(df) == 0 || ncol(df) <= 2) {
      cat(glue("⚠️ {fn_name} returned no useful data for season 2025, ",
               "skipping save.\n"))
      return(NULL)
    }
    write_csv(df, file_path)
    cat(glue("💾 Saved {fn_name} for season 2025\n"))
  }, error = function(e) {
    cat(glue("❌ Failed {fn_name} for season 2025: {e$message}\n"))
  })
}

# --- 5. Loop through game IDs for per-game hustle/defense
for (game_id in game_ids) {
  cat(glue("\n📦 Fetching game-based data for game: {game_id}\n"))
  fetch_and_save_game(game_id, "hustlestatsboxscore", nba_hustlestatsboxscore)
  fetch_and_save_game(game_id, "boxscoredefensivev2", nba_boxscoredefensivev2)
}

# --- 6. Fetch league-wide hustle team/player stats
cat(glue("\n📊 Fetching league-wide hustle stats for season 2025...\n"))
fetch_and_save_season("leaguehustlestatsplayer", nba_leaguehustlestatsplayer)
fetch_and_save_season("leaguehustlestatsteam", nba_leaguehustlestatteam)

cat("🎯 Hustle & defensive data collection complete.\n")
