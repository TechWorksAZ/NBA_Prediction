"""
Fetch advanced NBA team stats using nba_api.
This script fetches advanced metrics like offensive rating, defensive rating, etc.
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import teamdashboardbygeneralsplits
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
ADVANCED_DIR = RAW_DIR / "advanced"
SEASON = "2024-25"  # Current season

def get_team_ids() -> List[Dict]:
    """Get list of active NBA teams."""
    return teams.get_teams()

def fetch_team_advanced_stats(team_id: int, season: str) -> pd.DataFrame:
    """
    Fetch advanced stats for a specific team and season.
    
    Args:
        team_id: NBA team ID
        season: Season in format "YYYY-YY"
        
    Returns:
        DataFrame with advanced stats
    """
    try:
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        # Fetch advanced stats
        stats = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
            team_id=team_id,
            season=season,
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Advanced'
        ).get_data_frames()
        
        # Get overall stats
        overall_stats = stats[0]
        
        # Add team ID
        overall_stats['TEAM_ID'] = team_id
        
        logger.info(f"Fetched advanced stats for team {team_id}")
        return overall_stats
        
    except Exception as e:
        logger.error(f"Error fetching advanced stats for team {team_id}: {str(e)}")
        return pd.DataFrame()

def process_advanced_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw advanced stats into clean format.
    
    Args:
        df: Raw advanced stats DataFrame
        
    Returns:
        Cleaned DataFrame with advanced metrics
    """
    # Select relevant columns
    cols = [
        'TEAM_ID', 'GP', 'W', 'L', 'W_PCT',
        'OFF_RATING', 'DEF_RATING', 'NET_RATING',
        'AST_PCT', 'AST_TO', 'AST_RATIO',
        'OREB_PCT', 'DREB_PCT', 'REB_PCT',
        'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT',
        'PACE', 'PIE'
    ]
    
    df = df[cols].copy()
    
    # Clean up column names
    df.columns = df.columns.str.lower()
    
    # Add team abbreviation
    team_dict = {team['id']: team['abbreviation'] for team in get_team_ids()}
    df['team'] = df['team_id'].map(team_dict)
    
    return df

def main():
    """Main function to fetch and save advanced stats."""
    try:
        logger.info("Starting advanced stats fetch pipeline...")
        
        # Create output directory
        ADVANCED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Get all teams
        nba_teams = get_team_ids()
        
        # Initialize empty list for all advanced stats
        all_stats = []
        
        # Fetch advanced stats for each team
        for team in nba_teams:
            stats_df = fetch_team_advanced_stats(team['id'], SEASON)
            if not stats_df.empty:
                all_stats.append(stats_df)
                    
        # Combine all advanced stats
        if all_stats:
            combined_stats = pd.concat(all_stats, ignore_index=True)
            
            # Process advanced stats
            processed_stats = process_advanced_stats(combined_stats)
            
            # Save to CSV
            output_path = ADVANCED_DIR / "nba_advanced_stats.csv"
            processed_stats.to_csv(output_path, index=False)
            logger.info(f"Saved advanced stats to {output_path}")
            
        else:
            logger.warning("No advanced stats were fetched")
            
    except Exception as e:
        logger.error(f"Error in advanced stats pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 