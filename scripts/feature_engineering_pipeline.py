"""
NBA Prediction Model - Feature Engineering Pipeline

This script orchestrates the creation of all features needed for the NBA prediction model,
including player game features, team game features, betting features, and target features.
It combines the best aspects of both feature_engineering.py and run_feature_engineering.py.

Author: NBA Prediction Team
Date: 2024
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict
import os

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.features.player_game_features import create_player_game_features
from scripts.features.team_game_features import create_team_game_features
from scripts.features.betting_features import create_betting_features
from scripts.features.target_features import create_target_features
from scripts.features.shotchart_features import create_shotchart_features
from scripts.features.advanced_features import create_advanced_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_engineering.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_column_mapping(data_dir: Path) -> Dict:
    """Load the column mapping file."""
    mapping_file = data_dir / "data/processed/columns/FINAL_col_mapping.json"
    if not mapping_file.exists():
        logger.warning(f"Column mapping file not found: {mapping_file}")
        return {}
    with open(mapping_file, 'r') as f:
        return json.load(f)

def load_data(data_dir: str = "C:/Projects/NBA_Prediction") -> Dict[str, pd.DataFrame]:
    """Load all required data files with column mapping support."""
    logger.info("Loading data files...")
    data = {}
    data_dir = Path(data_dir)
    
    # Try to load column mapping
    mapping = load_column_mapping(data_dir)
    processed_data = mapping.get('processed_data', {})
    
    # Define file paths with fallback to direct paths
    file_paths = {
        'pg_data': processed_data.get('pg_data', {}).get('path', 'data/processed/pg_data.csv'),
        'pg_matchup': processed_data.get('pg_matchup', {}).get('path', 'data/processed/pg_matchup.csv'),
        'pg_pbp': processed_data.get('pg_pbp', {}).get('path', 'data/processed/pg_pbp.csv'),
        'pg_shotchart_detail': processed_data.get('pg_shotchart_detail', {}).get('path', 'data/processed/pg_shotchart_detail.csv'),
        'pg_shotchart_averages': processed_data.get('pg_shotchart_averages', {}).get('path', 'data/processed/pg_shotchart_averages.csv'),
        'tg_data': processed_data.get('tg_data', {}).get('path', 'data/processed/tg_data.csv'),
        'tg_starters': processed_data.get('tg_starters', {}).get('path', 'data/processed/tg_starter.csv'),
        'tg_bench': processed_data.get('tg_bench', {}).get('path', 'data/processed/tg_bench.csv'),
        'validated_odds': processed_data.get('validated_odds', {}).get('path', 'data/processed/validated_odds.csv'),
        'validated_games': processed_data.get('validated_games', {}).get('path', 'data/processed/validated_games.csv'),
        'tt_clutch': processed_data.get('tt_clutch', {}).get('path', 'data/processed/tt_clutch.csv'),
        'tt_data': processed_data.get('tt_data', {}).get('path', 'data/processed/tt_data.csv')
    }
    
    # Load each file
    for key, file_path in file_paths.items():
        full_path = data_dir / file_path
        logger.info(f"Loading {full_path}...")
        if full_path.exists():
            data[key] = pd.read_csv(full_path, low_memory=False)
        else:
            logger.warning(f"File not found: {full_path}")
            data[key] = pd.DataFrame()
    
    return data

def create_shot_chart_visualizations(data: Dict[str, pd.DataFrame], output_dir: Path):
    """Create shot chart visualizations for each game."""
    logger.info("Creating shot chart visualizations...")
    
    # Create output directory if it doesn't exist
    vis_dir = output_dir / "shotchartvisual"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique game IDs
    game_ids = data['pg_shotchart_detail']['GAME_ID'].unique()
    
    for game_id in game_ids:
        # Filter shot data for this game
        game_shots = data['pg_shotchart_detail'][data['pg_shotchart_detail']['GAME_ID'] == game_id]
        
        # Create figure
        plt.figure(figsize=(12, 11))
        
        # Create scatter plot
        made_shots = game_shots[game_shots['SHOT_MADE_FLAG'] == 1]
        missed_shots = game_shots[game_shots['SHOT_MADE_FLAG'] == 0]
        
        # Plot made shots in green
        plt.scatter(made_shots['LOC_X'], made_shots['LOC_Y'], 
                   c='green', alpha=0.6, label='Made Shots')
        
        # Plot missed shots in red
        plt.scatter(missed_shots['LOC_X'], missed_shots['LOC_Y'], 
                   c='red', alpha=0.6, label='Missed Shots')
        
        # Add court lines
        plt.plot([-250, 250], [0, 0], 'k-')  # Half court line
        plt.plot([-250, 250], [422.5, 422.5], 'k-')  # End line
        plt.plot([-250, -250], [0, 422.5], 'k-')  # Left sideline
        plt.plot([250, 250], [0, 422.5], 'k-')  # Right sideline
        
        # Add 3-point line
        three_pt_radius = 23.75
        three_pt_center = (0, 0)
        three_pt_arc = plt.Circle(three_pt_center, three_pt_radius, 
                                 fill=False, color='k')
        plt.gca().add_patch(three_pt_arc)
        
        # Add title and labels
        plt.title(f'Shot Chart - Game ID: {game_id}')
        plt.xlabel('X Location (feet)')
        plt.ylabel('Y Location (feet)')
        plt.legend()
        
        # Save figure with just the game ID as filename
        plt.savefig(vis_dir / f'{game_id}.png')
        plt.close()
    
    logger.info(f"Created shot chart visualizations in {vis_dir}")

def validate_game_ids(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Validate and align GAME_IDs across all feature sets.
    
    Matches validated_odds data with other datasets using GAME_ID and team information.
    """
    logger.info("Validating GAME_IDs across feature sets...")
    
    # Get the validated_odds data
    odds_data = data['validated_odds'].copy()
    games_data = data['validated_games'].copy()
    
    # Create match keys for games
    games_data['HOME_TEAM_COMBINED'] = (
        games_data['HOME_TEAM_CITY'].astype(str).str.lower().str.strip() + ' ' +
        games_data['HOME_TEAM_SLUG'].astype(str).str.lower().str.strip()
    )
    games_data['AWAY_TEAM_COMBINED'] = (
        games_data['AWAY_TEAM_CITY'].astype(str).str.lower().str.strip() + ' ' +
        games_data['AWAY_TEAM_SLUG'].astype(str).str.lower().str.strip()
    )
    games_data['HOME_TEAM_COMBINED'] = games_data['HOME_TEAM_COMBINED'].replace('portland blazers', 'portland trail blazers')
    games_data['HOME_TEAM_COMBINED'] = games_data['HOME_TEAM_COMBINED'].replace('philadelphia sixers', 'philadelphia 76ers')
    games_data['AWAY_TEAM_COMBINED'] = games_data['AWAY_TEAM_COMBINED'].replace('portland blazers', 'portland trail blazers')
    games_data['AWAY_TEAM_COMBINED'] = games_data['AWAY_TEAM_COMBINED'].replace('philadelphia sixers', 'philadelphia 76ers')
    games_data['match_key'] = games_data.apply(
        lambda row: f"{str(row['GAME_DATE']).strip().lower()}_{row['HOME_TEAM_COMBINED']}_{row['AWAY_TEAM_COMBINED']}",
        axis=1
    )

    # Create match keys for odds
    odds_data['HOME_TEAM_NORM'] = odds_data['HOME_TEAM'].astype(str).str.lower().str.strip()
    odds_data['AWAY_TEAM_NORM'] = odds_data['AWAY_TEAM'].astype(str).str.lower().str.strip()
    odds_data['HOME_TEAM_NORM'] = odds_data['HOME_TEAM_NORM'].replace('la', 'la clippers')
    odds_data['HOME_TEAM_NORM'] = odds_data['HOME_TEAM_NORM'].replace('portland blazers', 'portland trail blazers')
    odds_data['HOME_TEAM_NORM'] = odds_data['HOME_TEAM_NORM'].replace('philadelphia sixers', 'philadelphia 76ers')
    odds_data['AWAY_TEAM_NORM'] = odds_data['AWAY_TEAM_NORM'].replace('la', 'la clippers')
    odds_data['AWAY_TEAM_NORM'] = odds_data['AWAY_TEAM_NORM'].replace('portland blazers', 'portland trail blazers')
    odds_data['AWAY_TEAM_NORM'] = odds_data['AWAY_TEAM_NORM'].replace('philadelphia sixers', 'philadelphia 76ers')
    odds_data['match_key'] = odds_data.apply(
        lambda row: f"{str(row['GAME_DATE']).strip().lower()}_{row['HOME_TEAM_NORM']}_{row['AWAY_TEAM_NORM']}",
        axis=1
    )

    # Map GAME_IDs
    game_id_map = dict(zip(games_data['match_key'], games_data['GAME_ID']))
    odds_data['GAME_ID'] = odds_data['match_key'].map(game_id_map)

    # Create a DataFrame with all valid GAME_IDs
    valid_games = pd.DataFrame({
        'GAME_ID': sorted(list(set(odds_data['GAME_ID'].dropna()))),
        'MAPPED_ID': sorted(list(set(odds_data['GAME_ID'].dropna())))
    })
    
    logger.info(f"Created mapping for {len(valid_games)} games")
    
    # Log any unmapped games
    unmapped_odds = odds_data[odds_data['GAME_ID'].isna()]
    if not unmapped_odds.empty:
        logger.warning(f"Found {len(unmapped_odds)} unmapped games in odds data")
        logger.warning("Sample of unmapped games:")
        logger.warning(unmapped_odds[['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'match_key']].head())
    
    return valid_games

def create_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create team-specific features for both home and away teams."""
    logger.info("Creating team features...")
    
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Define the base columns we want to use, excluding score columns
    base_columns = [
        'TEAM_ID', 'TEAM_NAME', 'GAME_DATE',
        'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
        'PLUS_MINUS', 'POSS', 'PACE', 'OFF_RATING', 'DEF_RATING',
        'NET_RATING', 'AST_PCT', 'AST_TO', 'AST_RATIO', 'OREB_PCT',
        'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT',
        'PACE_PER40', 'PIE', 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK',
        'MIN_RANK', 'OFF_RATING_RANK', 'DEF_RATING_RANK', 'NET_RATING_RANK',
        'AST_PCT_RANK', 'AST_TO_RANK', 'AST_RATIO_RANK', 'OREB_PCT_RANK',
        'DREB_PCT_RANK', 'REB_PCT_RANK', 'TM_TOV_PCT_RANK', 'EFG_PCT_RANK',
        'TS_PCT_RANK', 'PACE_RANK', 'PIE_RANK', 'GP', 'W', 'L', 'W_PCT',
        'MIN', 'OFF_RATING_AVG', 'DEF_RATING_AVG', 'NET_RATING_AVG',
        'AST_PCT_AVG', 'AST_TO_AVG', 'AST_RATIO_AVG', 'OREB_PCT_AVG',
        'DREB_PCT_AVG', 'REB_PCT_AVG', 'TM_TOV_PCT_AVG', 'EFG_PCT_AVG',
        'TS_PCT_AVG', 'PACE_AVG', 'PIE_AVG'
    ]
    
    # Create home team features
    home_columns = {col: f'HOME_{col}' for col in base_columns}
    df = df.rename(columns=home_columns)
    
    # Create away team features using the same data
    away_columns = {col: f'AWAY_{col}' for col in base_columns}
    df = df.rename(columns=away_columns)
    
    # Ensure GAME_ID is preserved
    if 'GAME_ID' not in df.columns:
        logger.warning("GAME_ID column not found in team features")
        # Try to find it in the original data
        if 'GAME_ID' in df.columns:
            df['GAME_ID'] = df['GAME_ID']
        else:
            # If not found, try to extract it from other columns
            for col in df.columns:
                if 'GAME_ID' in col:
                    df['GAME_ID'] = df[col]
                    break
    
    return df

def save_features(features: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save all feature DataFrames to CSV files."""
    logger.info("Saving feature files...")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each feature DataFrame
    for name, df in features.items():
        if not df.empty:
            # Fix GAME_ID column to be int (or string if NaN)
            if 'GAME_ID' in df.columns:
                # If all GAME_IDs are numeric, cast to int; else, cast to string
                if pd.api.types.is_numeric_dtype(df['GAME_ID']):
                    if df['GAME_ID'].isnull().any():
                        df['GAME_ID'] = df['GAME_ID'].astype('Int64').astype(str)
                    else:
                        df['GAME_ID'] = df['GAME_ID'].astype(int)
                else:
                    df['GAME_ID'] = df['GAME_ID'].astype(str)
            output_path = output_dir / f"{name}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {name} features to {output_path}")
        else:
            logger.warning(f"Skipping empty DataFrame: {name}")

def update_validated_odds():
    """Update validated_odds.csv with correct GAME_IDs and team names."""
    data_dir = Path('C:/Projects/NBA_Prediction/data/processed')
    games_df = pd.read_csv(data_dir / 'validated_games.csv')
    odds_df = pd.read_csv(data_dir / 'validated_odds.csv')

    # Create match keys for games
    games_df['HOME_TEAM_COMBINED'] = (
        games_df['HOME_TEAM_CITY'].astype(str).str.lower().str.strip() + ' ' +
        games_df['HOME_TEAM_SLUG'].astype(str).str.lower().str.strip()
    )
    games_df['AWAY_TEAM_COMBINED'] = (
        games_df['AWAY_TEAM_CITY'].astype(str).str.lower().str.strip() + ' ' +
        games_df['AWAY_TEAM_SLUG'].astype(str).str.lower().str.strip()
    )
    games_df['HOME_TEAM_COMBINED'] = games_df['HOME_TEAM_COMBINED'].replace('portland blazers', 'portland trail blazers')
    games_df['HOME_TEAM_COMBINED'] = games_df['HOME_TEAM_COMBINED'].replace('philadelphia sixers', 'philadelphia 76ers')
    games_df['AWAY_TEAM_COMBINED'] = games_df['AWAY_TEAM_COMBINED'].replace('portland blazers', 'portland trail blazers')
    games_df['AWAY_TEAM_COMBINED'] = games_df['AWAY_TEAM_COMBINED'].replace('philadelphia sixers', 'philadelphia 76ers')
    games_df['match_key'] = games_df.apply(
        lambda row: f"{str(row['GAME_DATE']).strip().lower()}_{row['HOME_TEAM_COMBINED']}_{row['AWAY_TEAM_COMBINED']}",
        axis=1
    )

    # Create match keys for odds
    odds_df['HOME_TEAM_NORM'] = odds_df['HOME_TEAM'].astype(str).str.lower().str.strip()
    odds_df['AWAY_TEAM_NORM'] = odds_df['AWAY_TEAM'].astype(str).str.lower().str.strip()
    odds_df['HOME_TEAM_NORM'] = odds_df['HOME_TEAM_NORM'].replace('la', 'la clippers')
    odds_df['HOME_TEAM_NORM'] = odds_df['HOME_TEAM_NORM'].replace('portland blazers', 'portland trail blazers')
    odds_df['HOME_TEAM_NORM'] = odds_df['HOME_TEAM_NORM'].replace('philadelphia sixers', 'philadelphia 76ers')
    odds_df['AWAY_TEAM_NORM'] = odds_df['AWAY_TEAM_NORM'].replace('la', 'la clippers')
    odds_df['AWAY_TEAM_NORM'] = odds_df['AWAY_TEAM_NORM'].replace('portland blazers', 'portland trail blazers')
    odds_df['AWAY_TEAM_NORM'] = odds_df['AWAY_TEAM_NORM'].replace('philadelphia sixers', 'philadelphia 76ers')
    odds_df['match_key'] = odds_df.apply(
        lambda row: f"{str(row['GAME_DATE']).strip().lower()}_{row['HOME_TEAM_NORM']}_{row['AWAY_TEAM_NORM']}",
        axis=1
    )

    # Map GAME_IDs
    game_id_map = dict(zip(games_df['match_key'], games_df['GAME_ID']))
    odds_df['GAME_ID'] = odds_df['match_key'].map(game_id_map)

    # Clean up columns
    odds_df = odds_df.drop('match_key', axis=1)
    odds_df['HOME_TEAM'] = odds_df['HOME_TEAM'].replace('LA', 'LA Clippers')
    odds_df['AWAY_TEAM'] = odds_df['AWAY_TEAM'].replace('LA', 'LA Clippers')
    odds_df['HOME_TEAM'] = odds_df['HOME_TEAM'].replace('Portland Blazers', 'Portland Trail Blazers')
    odds_df['AWAY_TEAM'] = odds_df['AWAY_TEAM'].replace('Portland Blazers', 'Portland Trail Blazers')
    odds_df['HOME_TEAM'] = odds_df['HOME_TEAM'].replace('Philadelphia sixers', 'Philadelphia 76ers')
    odds_df['AWAY_TEAM'] = odds_df['AWAY_TEAM'].replace('Philadelphia sixers', 'Philadelphia 76ers')
    # Filter out rows where HOME_SCORE and AWAY_SCORE are both 0
    if 'HOME_SCORE' in odds_df.columns and 'AWAY_SCORE' in odds_df.columns:
        odds_df = odds_df[~((odds_df['HOME_SCORE'] == 0) & (odds_df['AWAY_SCORE'] == 0))]
    odds_df.to_csv(data_dir / 'validated_odds.csv', index=False)

def main():
    """Main function to run the feature engineering pipeline."""
    logger.info("Starting feature engineering pipeline...")
    
    try:
        # Set paths
        data_dir = Path("C:/Projects/NBA_Prediction")
        output_dir = data_dir / "data/processed/features"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        data = load_data(str(data_dir))
        
        # Validate and align GAME_IDs
        valid_games = validate_game_ids(data)
        
        # Create features
        logger.info("Creating player game features...")
        player_features = create_player_game_features(data)
        # Map GAME_IDs if needed
        if 'MAPPED_ID' in valid_games.columns:
            player_features['GAME_ID'] = player_features['GAME_ID'].map(
                dict(zip(valid_games['GAME_ID'], valid_games['MAPPED_ID']))
            ).fillna(player_features['GAME_ID'])
        player_features = player_features[player_features['GAME_ID'].isin(valid_games['GAME_ID'])]
        
        logger.info("Creating team game features...")
        team_features = create_team_game_features(data)
        # Map GAME_IDs if needed
        if 'MAPPED_ID' in valid_games.columns:
            team_features['GAME_ID'] = team_features['GAME_ID'].map(
                dict(zip(valid_games['GAME_ID'], valid_games['MAPPED_ID']))
            ).fillna(team_features['GAME_ID'])
        team_features = team_features[team_features['GAME_ID'].isin(valid_games['GAME_ID'])]
        
        logger.info("Creating betting features...")
        betting_features = create_betting_features(data)
        # Map GAME_IDs if needed
        if 'MAPPED_ID' in valid_games.columns:
            betting_features['GAME_ID'] = betting_features['GAME_ID'].map(
                dict(zip(valid_games['GAME_ID'], valid_games['MAPPED_ID']))
            ).fillna(betting_features['GAME_ID'])
        betting_features = betting_features[betting_features['GAME_ID'].isin(valid_games['GAME_ID'])]
        
        logger.info("Creating target features...")
        target_features = create_target_features(data)
        # Map GAME_IDs if needed
        if 'MAPPED_ID' in valid_games.columns:
            target_features['GAME_ID'] = target_features['GAME_ID'].map(
                dict(zip(valid_games['GAME_ID'], valid_games['MAPPED_ID']))
            ).fillna(target_features['GAME_ID'])
        target_features = target_features[target_features['GAME_ID'].isin(valid_games['GAME_ID'])]
        
        logger.info("Creating shot chart features...")
        shotchart_features = create_shotchart_features(data, visualize=False)
        # Map GAME_IDs if needed
        if 'MAPPED_ID' in valid_games.columns:
            shotchart_features['GAME_ID'] = shotchart_features['GAME_ID'].map(
                dict(zip(valid_games['GAME_ID'], valid_games['MAPPED_ID']))
            ).fillna(shotchart_features['GAME_ID'])
        shotchart_features = shotchart_features[shotchart_features['GAME_ID'].isin(valid_games['GAME_ID'])]
        
        logger.info("Creating advanced features...")
        advanced_features = create_advanced_features(data)
        # Map GAME_IDs if needed
        if 'MAPPED_ID' in valid_games.columns:
            advanced_features['GAME_ID'] = advanced_features['GAME_ID'].map(
                dict(zip(valid_games['GAME_ID'], valid_games['MAPPED_ID']))
            ).fillna(advanced_features['GAME_ID'])
        advanced_features = advanced_features[advanced_features['GAME_ID'].isin(valid_games['GAME_ID'])]
        
        # Create team-specific features
        logger.info("Creating team-specific features...")
        team_specific_features = create_team_features(data['tg_data'])
        # Map GAME_IDs if needed
        if 'MAPPED_ID' in valid_games.columns:
            team_specific_features['GAME_ID'] = team_specific_features['GAME_ID'].map(
                dict(zip(valid_games['GAME_ID'], valid_games['MAPPED_ID']))
            ).fillna(team_specific_features['GAME_ID'])
        team_specific_features = team_specific_features[team_specific_features['GAME_ID'].isin(valid_games['GAME_ID'])]
        
        # Combine all features
        features = {
            'player_game': player_features,
            'team_game': team_features,
            'betting': betting_features,
            'target': target_features,
            'shotchart': shotchart_features,
            'advanced': advanced_features,
            'team_specific': team_specific_features
        }
        
        # Save features
        save_features(features, output_dir)
        
        # Create shot chart visualizations
        logger.info("Creating shot chart visualizations...")
        create_shot_chart_visualizations(data, output_dir)
        
        # Update validated_odds.csv
        update_validated_odds()
        
        logger.info("Feature engineering pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {str(e)}")
        logger.exception("Detailed error information:")
        raise

if __name__ == "__main__":
    main() 