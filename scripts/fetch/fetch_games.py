import os
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.static import teams
from nba_api.stats.endpoints import scoreboardv2
from pathlib import Path

# Constants
DATA_BASE_DIR = "C:/Projects/NBA_Prediction"
DATA_DIR = os.path.join(DATA_BASE_DIR, "data/raw/core")
GAMES_FILE = os.path.join(DATA_DIR, "nba_games.csv")

def get_team_id_map():
    """Get mapping of team names to IDs."""
    team_map = {}
    for team in teams.get_teams():
        team_map[team['full_name']] = team['id']
    return team_map

def fetch_games(days_ahead=7):
    """Fetch games for today and next X days."""
    print("🏀 Fetching NBA games...")
    
    # Get today's date and next X days
    today = datetime.now().date()
    end_date = today + timedelta(days=days_ahead)
    
    all_games = []
    current_date = today
    
    while current_date <= end_date:
        print(f"📅 Fetching games for {current_date}...")
        
        # Get scoreboard for the date
        scoreboard = scoreboardv2.ScoreboardV2(
            game_date=current_date.strftime('%m/%d/%Y'),
            league_id='00'
        )
        
        # Get game data
        games = scoreboard.get_data_frames()[0]
        
        if not games.empty:
            # Add date column
            games['date'] = current_date.strftime('%Y-%m-%d')
            
            # Add to all games
            all_games.append(games)
        
        current_date += timedelta(days=1)
    
    if not all_games:
        print("⚠️ No games found!")
        return None
    
    # Combine all games
    games_df = pd.concat(all_games, ignore_index=True)
    
    # Get team ID mapping
    team_map = get_team_id_map()
    
    # Add team IDs
    games_df['home_team_id'] = games_df['HOME_TEAM_NAME'].map(team_map)
    games_df['away_team_id'] = games_df['VISITOR_TEAM_NAME'].map(team_map)
    
    # Rename columns
    games_df = games_df.rename(columns={
        'GAME_ID': 'game_id',
        'HOME_TEAM_NAME': 'home_team',
        'VISITOR_TEAM_NAME': 'away_team',
        'GAME_STATUS_TEXT': 'status'
    })
    
    # Select and order columns
    games_df = games_df[[
        'game_id', 'date', 'home_team', 'away_team', 
        'home_team_id', 'away_team_id', 'status'
    ]]
    
    # Save to file
    print(f"💾 Saving games to {GAMES_FILE}")
    games_df.to_csv(GAMES_FILE, index=False)
    
    print(f"✅ Successfully fetched {len(games_df)} games")
    return games_df

if __name__ == "__main__":
    fetch_games() 