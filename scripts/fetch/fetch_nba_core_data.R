# scripts/fetch/fetch_nba_core_data.R

# 👇 Ensure R knows where your user-installed packages are
.libPaths(file.path(Sys.getenv("USERPROFILE"), "R", "win-library", paste(R.version$major, R.version$minor, sep=".")))

# Load required packages
library(hoopR)
library(readr)

# Ensure output folder exists
dir.create("data/raw", showWarnings = FALSE, recursive = TRUE)

# Fetch & save NBA core data
cat("📅 Loading NBA schedule...\n")
schedule <- load_nba_schedule()
write_csv(schedule, "data/raw/nba_schedule.csv")

cat("📊 Loading NBA team box scores...\n")
team_box <- load_nba_team_box()
write_csv(team_box, "data/raw/nba_team_box_scores.csv")

cat("🧍 Loading NBA player box scores...\n")
player_box <- load_nba_player_box()
write_csv(player_box, "data/raw/nba_player_box_scores.csv")

cat("🎮 Loading NBA play-by-play data (this may take a while)...\n")
pbp <- load_nba_pbp()
write_csv(pbp, "data/raw/nba_play_by_play.csv")

cat("✅ All core NBA data saved to data/raw/\n")
