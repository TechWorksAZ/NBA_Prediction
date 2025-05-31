"""
Data Preprocessing Script - Column Standardization

This script standardizes column names across all data files to ensure consistency
in feature engineering and model training.

Author: NBA Prediction Team
Date: 2024
"""

import logging
import os
from pathlib import Path
from typing import Dict

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define standard column mappings
GAME_COLUMNS = {
    'GAME_ID': 'GAME_ID',
    'GAME_DATE': 'GAME_DATE',
    'HOME_TEAM': 'HOME_TEAM',
    'AWAY_TEAM': 'AWAY_TEAM',
    'HOME_SCORE': 'HOME_SCORE',
    'AWAY_SCORE': 'AWAY_SCORE',
    'VENUE': 'VENUE',
    'CITY': 'CITY',
    'STATE': 'STATE',
    'STATUS': 'STATUS'
}

ODDS_COLUMNS = {
    'GAME_ID': 'GAME_ID',
    'GAME_DATE': 'GAME_DATE',
    'HOME_TEAM': 'HOME_TEAM',
    'AWAY_TEAM': 'AWAY_TEAM',
    'SPREAD': 'SPREAD',
    'TOTAL': 'TOTAL',
    'HOME_MONEYLINE': 'HOME_MONEYLINE',
    'AWAY_MONEYLINE': 'AWAY_MONEYLINE',
    'HOME_SPREAD_ODDS': 'HOME_SPREAD_ODDS',
    'AWAY_SPREAD_ODDS': 'AWAY_SPREAD_ODDS',
    'OVER_ODDS': 'OVER_ODDS',
    'UNDER_ODDS': 'UNDER_ODDS'
}

def standardize_game_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize game data column names."""
    # Map team abbreviations to full names if needed
    team_mapping = {
        'HOME_TEAM_ABBR': 'HOME_TEAM',
        'AWAY_TEAM_ABBR': 'AWAY_TEAM'
    }
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Rename columns according to standard mapping
    df = df.rename(columns={
        **GAME_COLUMNS,
        **team_mapping
    })
    
    # Ensure all required columns exist
    for col in GAME_COLUMNS.values():
        if col not in df.columns:
            logger.warning(f"Missing column {col} in game data")
    
    return df

def standardize_odds_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize odds data column names."""
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Map BETMGM columns to standard names
    column_mapping = {
        'FULL-GAME_BETMGM_SPREAD_HOME': 'SPREAD',
        'FULL-GAME_BETMGM_TOTAL': 'TOTAL',
        'FULL-GAME_BETMGM_MONEYLINE_HOME': 'HOME_MONEYLINE',
        'FULL-GAME_BETMGM_MONEYLINE_AWAY': 'AWAY_MONEYLINE',
        'FULL-GAME_BETMGM_SPREAD_HOME_ODDS': 'HOME_SPREAD_ODDS',
        'FULL-GAME_BETMGM_SPREAD_AWAY_ODDS': 'AWAY_SPREAD_ODDS',
        'FULL-GAME_BETMGM_TOTAL_OVER_ODDS': 'OVER_ODDS',
        'FULL-GAME_BETMGM_TOTAL_UNDER_ODDS': 'UNDER_ODDS'
    }
    
    # Rename columns according to standard mapping
    df = df.rename(columns=column_mapping)
    
    # Ensure all required columns exist
    for col in ODDS_COLUMNS.values():
        if col not in df.columns:
            logger.warning(f"Missing column {col} in odds data")
    
    return df

def save_standardized_data(df: pd.DataFrame, output_path: Path, data_type: str):
    """Save standardized data to disk."""
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved standardized {data_type} data to {output_path}")

def main():
    """Main function to standardize all data files."""
    logger.info("Starting data standardization...")
    
    # Define paths
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    standardized_dir = processed_dir / "standardized"
    
    # Create standardized directory if it doesn't exist
    standardized_dir.mkdir(parents=True, exist_ok=True)
    
    # Process game data
    games_path = processed_dir / "validated_games.csv"
    if games_path.exists():
        games_df = pd.read_csv(games_path)
        games_df = standardize_game_data(games_df)
        save_standardized_data(
            games_df,
            standardized_dir / "games.csv",
            "game"
        )
    
    # Process odds data
    odds_path = processed_dir / "validated_odds.csv"
    if odds_path.exists():
        odds_df = pd.read_csv(odds_path)
        odds_df = standardize_odds_data(odds_df)
        save_standardized_data(
            odds_df,
            standardized_dir / "odds.csv",
            "odds"
        )
    
    logger.info("Data standardization complete!")

if __name__ == "__main__":
    main() 