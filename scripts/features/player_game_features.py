"""
Player Game Feature Engineering Module

This module creates features from player game data including:
- Basic statistics
- Advanced metrics
- Rolling averages
- Matchup-specific features
- Rest and fatigue metrics
- Shot location metrics
- Clutch performance features

Author: NBA Prediction Team
Date: 2024
"""

import logging
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.features.column_mappings import PG_COLS, get_col

logger = logging.getLogger(__name__)

def create_basic_stats_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic statistical features from player game data."""
    logger.info("Creating basic stat features...")
    
    # Game score
    df['GAME_SCORE'] = get_col(df, PG_COLS['PTS']) + 0.4 * get_col(df, PG_COLS['FGM']) - 0.7 * get_col(df, PG_COLS['FGA']) - \
                      0.4 * (get_col(df, PG_COLS['FTA']) - get_col(df, PG_COLS['FTM'])) + 0.7 * get_col(df, PG_COLS['OREB']) + \
                      0.3 * get_col(df, PG_COLS['DREB']) + get_col(df, PG_COLS['STL']) + 0.7 * get_col(df, PG_COLS['AST']) + 0.7 * get_col(df, PG_COLS['BLK']) - \
                      0.4 * get_col(df, PG_COLS['PF']) - get_col(df, PG_COLS['TOV'])
    
    # Fill NaN values with 0 for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def create_advanced_stats_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced statistical features."""
    logger.info("Creating advanced stat features...")
    
    # Calculate advanced stats
    advanced_stats = pd.DataFrame(index=df.index)
    
    # True shooting percentage
    advanced_stats['TRUE_SHOOTING_PCT'] = df[PG_COLS['PTS']] / (2 * (df[PG_COLS['FGA']] + 0.44 * df[PG_COLS['FTA']]))
    
    # Effective field goal percentage
    advanced_stats['EFFECTIVE_FG_PCT'] = (df[PG_COLS['FGM']] + 0.5 * df[PG_COLS['FG3M']]) / df[PG_COLS['FGA']]
    
    # Usage rate
    advanced_stats['USAGE_RATE'] = (df[PG_COLS['FGA']] + 0.44 * df[PG_COLS['FTA']] + df[PG_COLS['TOV']]) * 100 / df[PG_COLS['POSSESSIONS']]
    
    # Assist to turnover ratio
    advanced_stats['AST_TO_TOV'] = df[PG_COLS['AST']] / df[PG_COLS['TOV']].replace(0, 1)
    
    # Points per possession
    advanced_stats['PTS_PER_POSS'] = df[PG_COLS['PTS']] / df[PG_COLS['POSSESSIONS']]
    
    # Concatenate all features at once
    df = pd.concat([df, advanced_stats], axis=1)
    
    return df

def create_matchup_features(df: pd.DataFrame, matchup_df: pd.DataFrame) -> pd.DataFrame:
    """Create features from player matchup data."""
    logger.info("Creating matchup features...")
    
    matchup_features = []
    
    for (game_id, player_id), group in matchup_df.groupby(['GAME_ID', 'PLAYER_ID']):
        # Calculate matchup statistics
        total_matchup_time = group['MATCHUP_MINUTES'].sum()
        total_possessions = group['PARTIAL_POSSESSIONS'].sum()
        total_switches = group['SWITCHES_ON'].sum()
        total_points = group['PLAYER_POINTS'].sum()
        total_team_points = group['TEAM_POINTS'].sum()
        total_assists = group['MATCHUP_ASSISTS'].sum()
        total_potential_assists = group['MATCHUP_POTENTIAL_ASSISTS'].sum()
        total_fg_made = group['MATCHUP_FIELD_GOALS_MADE'].sum()
        total_fg_attempted = group['MATCHUP_FIELD_GOALS_ATTEMPTED'].sum()
        total_3p_made = group['MATCHUP_THREE_POINTERS_MADE'].sum()
        total_3p_attempted = group['MATCHUP_THREE_POINTERS_ATTEMPTED'].sum()
        total_blocks = group['MATCHUP_BLOCKS'].sum()
        total_help_blocks = group['HELP_BLOCKS'].sum()
        total_turnovers = group['MATCHUP_TURNOVERS'].sum()
        total_shooting_fouls = group['SHOOTING_FOULS'].sum()
        
        # Calculate percentages
        fg_percentage = total_fg_made / total_fg_attempted if total_fg_attempted > 0 else 0
        three_pt_percentage = total_3p_made / total_3p_attempted if total_3p_attempted > 0 else 0
        points_per_possession = total_points / total_possessions if total_possessions > 0 else 0
        
        # Create feature row
        feature_row = {
            'GAME_ID': game_id,
            'PLAYER_ID': player_id,
            'TOTAL_MATCHUP_TIME': total_matchup_time,
            'TOTAL_POSSESSIONS': total_possessions,
            'TOTAL_SWITCHES': total_switches,
            'TOTAL_POINTS': total_points,
            'TOTAL_TEAM_POINTS': total_team_points,
            'TOTAL_ASSISTS': total_assists,
            'TOTAL_POTENTIAL_ASSISTS': total_potential_assists,
            'TOTAL_FG_MADE': total_fg_made,
            'TOTAL_FG_ATTEMPTED': total_fg_attempted,
            'TOTAL_3P_MADE': total_3p_made,
            'TOTAL_3P_ATTEMPTED': total_3p_attempted,
            'TOTAL_BLOCKS': total_blocks,
            'TOTAL_HELP_BLOCKS': total_help_blocks,
            'TOTAL_TURNOVERS': total_turnovers,
            'TOTAL_SHOOTING_FOULS': total_shooting_fouls,
            'FG_PERCENTAGE': fg_percentage,
            'THREE_PT_PERCENTAGE': three_pt_percentage,
            'POINTS_PER_POSSESSION': points_per_possession,
            'ASSIST_RATIO': total_assists / total_potential_assists if total_potential_assists > 0 else 0,
            'BLOCK_RATE': total_blocks / total_possessions if total_possessions > 0 else 0,
            'TURNOVER_RATE': total_turnovers / total_possessions if total_possessions > 0 else 0,
            'FOUL_RATE': total_shooting_fouls / total_possessions if total_possessions > 0 else 0
        }
        
        matchup_features.append(feature_row)
    
    # Convert to DataFrame
    matchup_features_df = pd.DataFrame(matchup_features)
    
    # Merge with main dataframe
    df = df.merge(matchup_features_df, on=['GAME_ID', 'PLAYER_ID'], how='left')
    
    # Fill NaN values with 0 for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling average features."""
    logger.info("Creating rolling features...")
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    stat_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 
                 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'TRUE_SHOOTING_PCT', 'EFFECTIVE_FG_PCT']

    # Create a dictionary to store all rolling features
    rolling_features = {}
    
    for window in [3, 5, 10]:
        for col in stat_cols:
            if col in df.columns:
                # Calculate rolling mean and std
                rolling_mean = df.groupby('PLAYER_ID')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
                rolling_std = df.groupby('PLAYER_ID')[col].transform(lambda x: x.rolling(window, min_periods=1).std())
                
                # Store in dictionary
                rolling_features[f'{col}_ROLLING_{window}'] = rolling_mean
                rolling_features[f'{col}_ROLLING_{window}_STD'] = rolling_std
    
    # Convert dictionary to DataFrame and concatenate with original
    rolling_df = pd.DataFrame(rolling_features, index=df.index)
    df = pd.concat([df, rolling_df], axis=1)

    return df

def create_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rest-related features."""
    logger.info("Creating rest features...")
    
    # Ensure GAME_DATE is datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Create a dictionary to store rest features
    rest_features = {}
    
    # Calculate days rest between games
    rest_features['DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days
    
    # Create back-to-back flag
    rest_features['IS_BACK_TO_BACK'] = (rest_features['DAYS_REST'] == 1).astype(int)
    
    # Create rest categories
    rest_features['REST_CATEGORY'] = pd.cut(
        rest_features['DAYS_REST'],
        bins=[-float('inf'), 0, 1, 2, 3, float('inf')],
        labels=['0', '1', '2', '3', '4+'],
        right=False
    )
    
    # Convert dictionary to DataFrame and concatenate with original
    rest_df = pd.DataFrame(rest_features, index=df.index)
    df = pd.concat([df, rest_df], axis=1)
    
    return df

def create_shot_location_features(df: pd.DataFrame, shotchart_df: pd.DataFrame) -> pd.DataFrame:
    """Create shot location features."""
    logger.info("Creating shot location features...")

    shot_zone_stats = shotchart_df.groupby(['GAME_ID', 'PLAYER_ID', 'SHOT_ZONE_BASIC'])['SHOT_MADE_FLAG'].agg(['count', 'mean']).unstack().fillna(0)
    shot_zone_stats.columns = [f'SHOT_ZONE_{col[0]}_{col[1].lower()}' for col in shot_zone_stats.columns]

    df = df.merge(shot_zone_stats, left_on=['GAME_ID', 'PLAYER_ID'], right_on=['GAME_ID', 'PLAYER_ID'], how='left')

    return df

def create_clutch_features(df: pd.DataFrame, clutch_df: pd.DataFrame) -> pd.DataFrame:
    """Create clutch performance features."""
    logger.info("Creating clutch features...")
    
    # Merge clutch stats
    df = df.merge(
        clutch_df[['TEAM_ID', 'GP', 'W', 'L', 'W_PCT', 'PTS', 'PLUS_MINUS']],
        left_on='PLAYER_TEAM_ID',
        right_on='TEAM_ID',
        how='left'
    )
    
    # Rename clutch columns
    clutch_cols = {
        'GP': 'CLUTCH_GP',
        'W': 'CLUTCH_W',
        'L': 'CLUTCH_L',
        'W_PCT': 'CLUTCH_W_PCT',
        'PTS': 'CLUTCH_PTS',
        'PLUS_MINUS': 'CLUTCH_PLUS_MINUS'
    }
    df = df.rename(columns=clutch_cols)
    
    return df

def create_player_game_features(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create all player game features."""
    logger.info("Starting player game feature engineering...")
    
    # Get player game data
    pg_df = data['pg_data']
    
    # Merge scores from validated_games.csv
    games_path = Path('data/processed/validated_games.csv')
    games = pd.read_csv(games_path, usecols=['GAME_ID', 'HOME_SCORE', 'AWAY_SCORE'])
    pg_df = pg_df.merge(games, on='GAME_ID', how='left')
    
    # Create basic stat features
    df = create_basic_stats_features(pg_df)
    
    # Create advanced stat features
    df = create_advanced_stats_features(df)
    
    # Create matchup features
    df = create_matchup_features(df, data['pg_matchup'])
    
    # Create rolling features
    df = create_rolling_features(df)
    
    # Create rest features
    df = create_rest_features(df)
    
    # Create shot location features if available
    if 'pg_shotchart' in data:
        df = create_shot_location_features(df, data['pg_shotchart'])
    
    # Create clutch features if available
    if 'tt_clutch' in data:
        df = create_clutch_features(df, data['tt_clutch'])
    
    # Sort by game and player
    df = df.sort_values(['GAME_ID', 'PLAYER_ID'])
    
    # Drop duplicate columns (those ending with .1, .2, etc.)
    duplicate_cols = [col for col in df.columns if col.endswith(('.1', '.2', '.3', '.4', '.5'))]
    if duplicate_cols:
        logger.info(f"Dropping duplicate columns: {duplicate_cols}")
        df = df.drop(columns=duplicate_cols)
    
    # Ensure no duplicate column names remain
    df = df.loc[:, ~df.columns.duplicated()]
    
    logger.info(f"Created player game features with shape {df.shape}")
    return df
