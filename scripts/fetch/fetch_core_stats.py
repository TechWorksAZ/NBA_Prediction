"""
Fetch core NBA team stats using nba_api.
This script fetches basic team stats like points, rebounds, assists, etc.
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import teamgamelogs
from nba_api.stats.static import teams
from pathlib import Path
import logging
from datetime import datetime, timedelta
import time
from typing import List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RAW_DIR = Path("C:/Projects/NBA_Prediction/data/raw")
CORE_DIR = RAW_DIR / "core"
SEASON = "2024-25"  # Current season

def get_team_ids() -> List[Dict]:
    """Get list of active NBA teams."""
    return teams.get_teams()

def fetch_team_game_logs(team_id: int, season: str) -> pd.DataFrame:
    """
    Fetch game logs for a specific team and season.
    
    Args:
        team_id: NBA team ID
        season: Season in format "YYYY-YY"
        
    Returns:
        DataFrame with game stats
    """
    try:
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        # Fetch game logs
        logs = teamgamelogs.TeamGameLogs(
            team_id_nullable=team_id,
            season_nullable=season
        ).get_data_frames()[0]
        
        logger.info(f"Fetched game logs for team {team_id}")
        return logs
        
    except Exception as e:
        logger.error(f"Error fetching game logs for team {team_id}: {str(e)}")
        return pd.DataFrame()

def process_game_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw game logs into clean format.
    
    Args:
        df: Raw game logs DataFrame
        
    Returns:
        Cleaned DataFrame with core stats
    """
    # Select relevant columns
    cols = [
        'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_ABBREVIATION',
        'MATCHUP', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
        'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
        'PLUS_MINUS'
    ]
    
    df = df[cols].copy()
    
    # Clean up column names
    df.columns = df.columns.str.lower()
    
    # Convert date
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Extract home/away and opponent from matchup
    df['is_home'] = df['matchup'].str.contains('vs')
    df['opponent'] = df['matchup'].str.extract(r'(vs|@)\s+([A-Z]+)', expand=True)[1]
    
    return df

def main():
    """Main function to fetch and save core stats."""
    try:
        logger.info("Starting core stats fetch pipeline...")
        
        # Create output directory
        CORE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Get all teams
        nba_teams = get_team_ids()
        
        # Initialize empty list for all game logs
        all_logs = []
        
        # Fetch game logs for each team
        for team in nba_teams:
            logs_df = fetch_team_game_logs(team['id'], SEASON)
            if not logs_df.empty:
                all_logs.append(logs_df)
                    
        # Combine all game logs
        if all_logs:
            combined_logs = pd.concat(all_logs, ignore_index=True)
            
            # Process game logs
            processed_logs = process_game_logs(combined_logs)
            
            # Save to CSV
            output_path = CORE_DIR / "nba_team_stats.csv"
            processed_logs.to_csv(output_path, index=False)
            logger.info(f"Saved core stats to {output_path}")
            
        else:
            logger.warning("No game logs were fetched")
            
    except Exception as e:
        logger.error(f"Error in core stats pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 