"""
Shot Chart Feature Engineering Module

This module creates features from shot chart data including:
- Shot location metrics
- Shot distance analysis
- Shot zone efficiency
- Shot chart visualizations

Author: NBA Prediction Team
Date: 2024
"""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def bin_shot_distance(df: pd.DataFrame) -> pd.DataFrame:
    if 'SHOT_DISTANCE' in df.columns:
        df['DISTANCE_BIN'] = pd.cut(
            df['SHOT_DISTANCE'],
            bins=[0, 5, 10, 15, 20, 25, 30, np.inf],
            labels=['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']
        )
    return df

def create_shot_location_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating shot location features...")

    if 'SHOT_ZONE_BASIC' in df.columns:
        zone_counts = df.groupby(['GAME_ID', 'PLAYER_ID', 'SHOT_ZONE_BASIC']).size().unstack(fill_value=0)
        zone_pcts = zone_counts.div(zone_counts.sum(axis=1), axis=0)
        zone_pcts.columns = [f'SHOT_ZONE_{col.upper()}_PCT' for col in zone_pcts.columns]
        df = pd.merge(df, zone_pcts, on=['GAME_ID', 'PLAYER_ID'], how='left')

    if 'SHOT_DISTANCE' in df.columns:
        df['AVG_SHOT_DISTANCE'] = df.groupby(['GAME_ID', 'PLAYER_ID'])['SHOT_DISTANCE'].transform('mean')
        df['SHOT_DISTANCE_STD'] = df.groupby(['GAME_ID', 'PLAYER_ID'])['SHOT_DISTANCE'].transform('std')
        df['LOG_SHOT_DISTANCE'] = np.log1p(df['SHOT_DISTANCE'])
        df['CLIPPED_SHOT_DISTANCE'] = df['SHOT_DISTANCE'].clip(upper=35)
        df = bin_shot_distance(df)

    return df

def create_shot_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create shot efficiency features."""
    logger.info("Creating shot efficiency features...")

    # Calculate effective field goal percentage by distance
    df['EFFECTIVE_FG'] = df['SHOT_MADE_FLAG'] * (1 + 0.5 * (df['SHOT_TYPE'] == '3PT'))
    
    # Group by distance bins
    dist_fg = df.groupby(['GAME_ID', 'PLAYER_ID', 'DISTANCE_BIN'], observed=True)['SHOT_MADE_FLAG'].mean().unstack()
    dist_fg.columns = [f'FG_PCT_{col}' for col in dist_fg.columns]
    
    # Calculate effective field goal percentage by distance
    efg_dist = df.groupby(['GAME_ID', 'PLAYER_ID', 'DISTANCE_BIN'], observed=True)['EFFECTIVE_FG'].mean().unstack()
    efg_dist.columns = [f'EFG_PCT_{col}' for col in efg_dist.columns]
    
    # Merge features
    df = df.merge(dist_fg, on=['GAME_ID', 'PLAYER_ID'], how='left')
    df = df.merge(efg_dist, on=['GAME_ID', 'PLAYER_ID'], how='left')

    return df

def create_shot_visualization(df: pd.DataFrame, output_dir: Path, game_id: str, player_id: str) -> None:
    logger.info(f"Creating shot chart visualization for game {game_id}, player {player_id}...")
    game_data = df[(df['GAME_ID'] == game_id) & (df['PLAYER_ID'] == player_id)]

    if game_data.empty:
        logger.warning(f"No shot data found for game {game_id}, player {player_id}")
        return

    plt.figure(figsize=(12, 8))
    plt.scatter(
        game_data['LOC_X'],
        game_data['LOC_Y'],
        c=game_data['SHOT_MADE_FLAG'],
        cmap='coolwarm',
        alpha=0.6,
        edgecolors='k',
        marker='o'
    )

    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"Shot Chart - Game {game_id}, Player {player_id}")
    plt.xlabel("Court X Position")
    plt.ylabel("Court Y Position")

    output_path = output_dir / f"shotchart_{game_id}_{player_id}.png"
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved shot chart visualization to {output_path}")

def create_shotchart_features(data: Dict[str, pd.DataFrame], visualize: bool = False) -> pd.DataFrame:
    """Create shot chart features.
    
    Args:
        data: Dictionary containing DataFrames
        visualize: Whether to create shot chart visualizations (default: False)
    
    Returns:
        DataFrame with shot chart features
    """
    logger.info("Starting shot chart feature engineering...")
    shotchart_df = data['pg_shotchart_detail']

    df = create_shot_location_features(shotchart_df)
    df = create_shot_efficiency_features(df)

    if visualize:
        logger.info("Creating shot chart visualizations...")
        output_dir = Path('data/processed/engineered/shotchart_visuals')
        output_dir.mkdir(parents=True, exist_ok=True)

        for game_id in df['GAME_ID'].unique():
            for player_id in df[df['GAME_ID'] == game_id]['PLAYER_ID'].unique():
                create_shot_visualization(df, output_dir, game_id, player_id)
    else:
        logger.info("Skipping shot chart visualizations (visualize=False)")

    logger.info("Completed shot chart feature engineering")
    return df
