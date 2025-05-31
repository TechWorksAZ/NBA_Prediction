"""
Team Game Feature Engineering Module

This module creates features from team game data including:
- Basic team statistics
- Advanced team metrics
- Rolling averages
- Lineup-specific features
- Rest and fatigue metrics

Author: NBA Prediction Team
Date: 2024
"""

import logging
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.features.column_mappings import TG_COLS, TG_STARTER_COLS, TG_BENCH_COLS, get_col

logger = logging.getLogger(__name__)

def create_basic_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic team-level features."""
    logger.info("Creating basic team features...")

    # Ensure all relevant columns are numeric and handle missing values
    for col in TG_COLS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            logger.warning(f"Missing column {col} in team data")
            df[col] = 0

    # Shooting efficiency with safe division
    df['TRUE_SHOOTING_PCT'] = df[TG_COLS['POINTS']] / (2 * (df[TG_COLS['FIELD_GOALS_ATTEMPTED']] + 0.44 * df[TG_COLS['FREE_THROWS_ATTEMPTED']]).replace(0, 1))
    df['EFFECTIVE_FG_PCT'] = (df[TG_COLS['FIELD_GOALS_MADE']] + 0.5 * df[TG_COLS['THREE_POINTERS_MADE']]) / df[TG_COLS['FIELD_GOALS_ATTEMPTED']].replace(0, 1)
    
    # Four factors with safe division
    df['EFG_PCT'] = (df[TG_COLS['FIELD_GOALS_MADE']] + 0.5 * df[TG_COLS['THREE_POINTERS_MADE']]) / df[TG_COLS['FIELD_GOALS_ATTEMPTED']].replace(0, 1)
    df['TOV_PCT'] = df[TG_COLS['TURNOVERS']] / (df[TG_COLS['FIELD_GOALS_ATTEMPTED']] + 0.44 * df[TG_COLS['FREE_THROWS_ATTEMPTED']] + df[TG_COLS['TURNOVERS']]).replace(0, 1)
    df['OREB_PCT'] = df[TG_COLS['REBOUNDS_OFFENSIVE']] / (df[TG_COLS['REBOUNDS_OFFENSIVE']] + df[TG_COLS['REBOUNDS_DEFENSIVE']]).replace(0, 1)
    df['FT_RATE'] = df[TG_COLS['FREE_THROWS_MADE']] / df[TG_COLS['FIELD_GOALS_ATTEMPTED']].replace(0, 1)
    
    # Pace with safe division
    df['PACE'] = 48 * ((df[TG_COLS['FIELD_GOALS_ATTEMPTED']] + 0.44 * df[TG_COLS['FREE_THROWS_ATTEMPTED']] - df[TG_COLS['REBOUNDS_OFFENSIVE']] + df[TG_COLS['TURNOVERS']]) / (2 * (df[TG_COLS['MINUTES']] / 5)).replace(0, 1))
    
    # Fill any remaining NaN values with 0
    df = df.fillna(0)
    
    return df

def create_advanced_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced team statistical features."""
    logger.info("Creating advanced team features...")
    
    # Create a dictionary to store advanced features
    advanced_features = {}
    
    # Offensive and defensive ratings with safe division
    possessions = df[TG_COLS['POSSESSIONS']].replace(0, 1)
    advanced_features['OFFENSIVE_RATING'] = 100 * (df[TG_COLS['POINTS']] / possessions)
    advanced_features['DEFENSIVE_RATING'] = 100 * (df[TG_COLS['POINTS']] / possessions)
    advanced_features['NET_RATING'] = advanced_features['OFFENSIVE_RATING'] - advanced_features['DEFENSIVE_RATING']
    
    # Assist and turnover ratios with safe division
    denominator = (df[TG_COLS['FIELD_GOALS_ATTEMPTED']] + 0.44 * df[TG_COLS['FREE_THROWS_ATTEMPTED']] + df[TG_COLS['TURNOVERS']]).replace(0, 1)
    advanced_features['ASSIST_RATIO'] = df[TG_COLS['ASSISTS']] / denominator
    advanced_features['TURNOVER_RATIO'] = df[TG_COLS['TURNOVERS']] / denominator
    
    # Rebounding percentages with safe division
    total_rebounds = (df[TG_COLS['REBOUNDS_OFFENSIVE']] + df[TG_COLS['REBOUNDS_DEFENSIVE']]).replace(0, 1)
    advanced_features['OREB_PCT'] = df[TG_COLS['REBOUNDS_OFFENSIVE']] / total_rebounds
    advanced_features['DREB_PCT'] = df[TG_COLS['REBOUNDS_DEFENSIVE']] / total_rebounds
    
    # Convert dictionary to DataFrame and concatenate with original
    advanced_df = pd.DataFrame(advanced_features, index=df.index)
    df = pd.concat([df, advanced_df], axis=1)
    
    # Fill any remaining NaN values with 0
    df = df.fillna(0)
    
    return df

def create_lineup_features(df: pd.DataFrame, starters_df: pd.DataFrame, bench_df: pd.DataFrame) -> pd.DataFrame:
    """Create lineup-specific features comparing starter and bench performance."""
    logger.info("Creating lineup features...")
    
    # Ensure required columns exist in starters and bench data
    for col in TG_STARTER_COLS.values():
        if col not in starters_df.columns:
            logger.warning(f"Missing column {col} in starters data")
            starters_df[col] = 0
    for col in TG_BENCH_COLS.values():
        if col not in bench_df.columns:
            logger.warning(f"Missing column {col} in bench data")
            bench_df[col] = 0
    
    # Calculate starter totals
    starter_totals = starters_df.groupby(['GAME_ID', 'TEAM_ID']).agg({
        'POINTS': 'sum',
        'MINUTES': 'sum'
    }).reset_index()
    starter_totals.columns = ['GAME_ID', 'TEAM_ID', 'STARTER_POINTS', 'STARTER_MINUTES']
    
    # Calculate bench totals
    bench_totals = bench_df.groupby(['GAME_ID', 'TEAM_ID']).agg({
        'POINTS': 'sum',
        'MINUTES': 'sum'
    }).reset_index()
    bench_totals.columns = ['GAME_ID', 'TEAM_ID', 'BENCH_POINTS', 'BENCH_MINUTES']
    
    # Merge totals with main dataframe
    df = df.merge(starter_totals, on=['GAME_ID', 'TEAM_ID'], how='left')
    df = df.merge(bench_totals, on=['GAME_ID', 'TEAM_ID'], how='left')
    
    # Fill missing values from merges
    df['STARTER_POINTS'] = df['STARTER_POINTS'].fillna(0)
    df['STARTER_MINUTES'] = df['STARTER_MINUTES'].fillna(0)
    df['BENCH_POINTS'] = df['BENCH_POINTS'].fillna(0)
    df['BENCH_MINUTES'] = df['BENCH_MINUTES'].fillna(0)
    
    # Convert columns to numeric types before calculations
    for col in ['STARTER_MINUTES', 'BENCH_MINUTES', 'STARTER_POINTS', 'BENCH_POINTS']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Calculate lineup ratios
    df['STARTER_BENCH_MINUTES_RATIO'] = df['STARTER_MINUTES'] / df['BENCH_MINUTES'].replace(0, 1)
    df['STARTER_BENCH_POINTS_RATIO'] = df['STARTER_POINTS'] / df['BENCH_POINTS'].replace(0, 1)
    
    # Drop intermediate columns
    df = df.drop(['STARTER_POINTS', 'STARTER_MINUTES', 'BENCH_POINTS', 'BENCH_MINUTES'], axis=1)
    
    # Fill any remaining NaN values with 0
    df = df.fillna(0)
    
    return df

def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling average features."""
    logger.info("Creating rolling features...")
    df = df.sort_values(['TEAM_ID', 'GAME_ID'])

    stat_cols = ['POINTS', 'REBOUNDS_TOTAL', 'ASSISTS', 'STEALS', 'BLOCKS', 'TURNOVERS',
                 'FIELD_GOALS_PERCENTAGE', 'THREE_POINTERS_PERCENTAGE', 'FREE_THROWS_PERCENTAGE',
                 'TRUE_SHOOTING_PCT', 'EFFECTIVE_FG_PCT', 'PACE']

    for window in [3, 5, 10]:
        for col in stat_cols:
            if col in df.columns:
                df[f'{col}_ROLLING_{window}'] = df.groupby('TEAM_ID')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
                df[f'{col}_ROLLING_{window}_STD'] = df.groupby('TEAM_ID')[col].transform(lambda x: x.rolling(window, min_periods=1).std())

    # Fill any remaining NaN values with 0
    df = df.fillna(0)
    
    return df

def create_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rest-related features for teams."""
    logger.info("Creating rest features...")

    # Remove GAME_DATE conversion and related calculations
    # df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
    # df['GAME_DATE'] = df['GAME_DATE'].fillna(pd.Timestamp('2024-01-01'))
    # df['DAYS_REST'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
    # df['IS_BACK_TO_BACK'] = (df['DAYS_REST'] == 1).astype(int)
    # df['REST_CATEGORY'] = pd.cut(
    #     df['DAYS_REST'],
    #     bins=[-float('inf'), 0, 1, 2, 3, float('inf')],
    #     labels=['0', '1', '2', '3', '4+'],
    #     right=False
    # )
    
    # Fill any remaining NaN values with 0
    df = df.fillna(0)
    
    return df

def create_team_game_features(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create all team game features."""
    logger.info("Starting team game feature engineering...")
    
    # Get team game data
    tg_df = data['tg_data'].copy()
    
    # Log available columns and sample data
    logger.info(f"Available columns in tg_data: {tg_df.columns.tolist()}")
    
    # Store team name columns and minutes
    team_name_cols = ['TEAM_NAME', 'TEAM_CITY', 'TEAM_ABBREVIATION', 'TEAM_SLUG']
    minutes_col = 'MINUTES'
    
    # Check which columns actually exist in the data
    existing_team_cols = [col for col in team_name_cols if col in tg_df.columns]
    if minutes_col in tg_df.columns:
        existing_team_cols.append(minutes_col)
    
    logger.info(f"Found existing team columns: {existing_team_cols}")
    
    # Store the original team name and minutes data
    team_data = tg_df[['GAME_ID', 'TEAM_ID'] + existing_team_cols].copy()
    
    # Rename MINUTES to TOTAL_GAME_MINUTES if it exists
    if minutes_col in team_data.columns:
        team_data = team_data.rename(columns={minutes_col: 'TOTAL_GAME_MINUTES'})
        # Update existing_team_cols to use the new name
        existing_team_cols = [col if col != minutes_col else 'TOTAL_GAME_MINUTES' for col in existing_team_cols]
    
    # Create a copy of tg_df without the team name columns for feature creation
    feature_df = tg_df.drop(columns=existing_team_cols, errors='ignore')
    
    # Create basic team features
    df = create_basic_team_features(feature_df)
    
    # Create advanced team features
    df = create_advanced_team_features(df)
    
    # Create lineup features
    df = create_lineup_features(df, data['tg_starters'], data['tg_bench'])
    
    # Create rolling features
    df = create_rolling_features(df)
    
    # Create rest features
    df = create_rest_features(df)
    
    # Sort by GAME_ID and TEAM_ID instead of GAME_DATE
    df = df.sort_values(['TEAM_ID', 'GAME_ID'])
    
    # Log shape before merge
    logger.info(f"Feature DataFrame shape before merge: {df.shape}")
    logger.info(f"Team data shape before merge: {team_data.shape}")
    
    # Drop any existing team columns from df to avoid duplicates
    df = df.drop(columns=existing_team_cols, errors='ignore')
    
    # Merge back the team name and minutes data
    df = df.merge(team_data, on=['GAME_ID', 'TEAM_ID'], how='left')
    
    # Reorder columns to put team name columns after HOME_TEAM_ID
    base_cols = ['GAME_ID', 'TEAM_ID', 'AWAY_TEAM_ID', 'HOME_TEAM_ID']
    other_cols = [col for col in df.columns if col not in base_cols + existing_team_cols]
    df = df[base_cols + existing_team_cols + other_cols]
    
    # Log final shape and columns
    logger.info(f"Created team game features with shape {df.shape}")
    logger.info(f"Final columns: {df.columns.tolist()}")
    
    # Log sample of team name columns
    team_cols = [col for col in df.columns if col in existing_team_cols]
    if team_cols:
        logger.info(f"Sample of final team name columns:\n{df[team_cols].head()}")
    
    missing_values = df.isnull().sum()
    if missing_values.any():
        logger.warning(f"Columns with missing values:\n{missing_values[missing_values > 0]}")
    
    return df
