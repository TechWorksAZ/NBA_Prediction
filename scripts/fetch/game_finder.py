import os
import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder
import logging
from typing import Optional
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_games(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    season: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch NBA game IDs and basic game info, filtering out preseason and G League games.
    Combines rows for each game into a single row with both teams' information.
    
    Args:
        date_from: Start date in YYYY-MM-DD format
        date_to: End date in YYYY-MM-DD format
        season: Season in YYYY-YY format (e.g., '2024-25')
    
    Returns:
        DataFrame with game IDs and basic game info
    """
    try:
        # Fetch games using NBA API
        gamefinder = LeagueGameFinder(
            date_from_nullable=date_from,
            date_to_nullable=date_to,
            season_nullable=season,
            league_id_nullable='00'  # NBA games only
        )
        games = gamefinder.get_data_frames()[0]
        
        # Log date range
        logger.info(f"Date range in returned data:")
        logger.info(f"Earliest date: {games['GAME_DATE'].min()}")
        logger.info(f"Latest date: {games['GAME_DATE'].max()}")
        logger.info(f"Total games before filtering: {len(games)}")
        
        # Filter out preseason games (001) and G League games (204)
        games = games[~games['GAME_ID'].str.startswith(('001', '204'))]
        
        # Keep only essential columns
        games = games[[
            'GAME_ID',
            'GAME_DATE',
            'TEAM_ID',
            'TEAM_ABBREVIATION',
            'TEAM_NAME',
            'MATCHUP'
        ]]
        
        # Split into home and away teams
        home_games = games[games['MATCHUP'].str.contains(' vs. ')].copy()
        away_games = games[games['MATCHUP'].str.contains(' @ ')].copy()
        
        # Rename columns for home team
        home_games.columns = ['GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'HOME_TEAM_ABBR', 'HOME_TEAM_NAME', 'MATCHUP']
        
        # Rename columns for away team
        away_games.columns = ['GAME_ID', 'GAME_DATE', 'AWAY_TEAM_ID', 'AWAY_TEAM_ABBR', 'AWAY_TEAM_NAME', 'MATCHUP']
        
        # Merge home and away games
        combined_games = pd.merge(
            home_games,
            away_games[['GAME_ID', 'AWAY_TEAM_ID', 'AWAY_TEAM_ABBR', 'AWAY_TEAM_NAME']],
            on='GAME_ID',
            how='inner'
        )
        
        # Drop the MATCHUP column as we'll create a new one
        combined_games = combined_games.drop('MATCHUP', axis=1)
        
        # Create a new MATCHUP column
        combined_games['MATCHUP'] = combined_games['HOME_TEAM_ABBR'] + ' vs. ' + combined_games['AWAY_TEAM_ABBR']
        
        # Reorder columns
        combined_games = combined_games[[
            'GAME_ID',
            'GAME_DATE',
            'HOME_TEAM_ID',
            'HOME_TEAM_ABBR',
            'HOME_TEAM_NAME',
            'AWAY_TEAM_ID',
            'AWAY_TEAM_ABBR',
            'AWAY_TEAM_NAME',
            'MATCHUP'
        ]]
        
        logger.info(f"Total games after combining: {len(combined_games)}")
        return combined_games
        
    except Exception as e:
        logger.error(f"Error fetching games: {str(e)}")
        raise

def main():
    """Main function to fetch and save game data."""
    # Set up paths
    base_dir = "C:/Projects/NBA_Prediction"
    raw_dir = os.path.join(base_dir, "data", "raw", "core")
    os.makedirs(raw_dir, exist_ok=True)
    output_file = os.path.join(raw_dir, "games.csv")
    
    try:
        # Get current season games
        games = get_games(season="2024-25")
        
        # Save to CSV
        games.to_csv(output_file, index=False)
        logger.info(f"Saved {len(games)} games to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 