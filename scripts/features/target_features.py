"""
Target Feature Engineering Module

This module creates target variables for model training including:
- Win/loss outcomes
- Spread margins
- Total points
- Betting outcomes (ATS, O/U)

Author: NBA Prediction Team
Date: 2024
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def create_game_outcome_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create game outcome features."""
    logger.info("Creating game outcome features...")
    
    # Win/loss outcome
    df['HOME_WIN'] = (df['HOME_SCORE'] > df['AWAY_SCORE']).astype(int)
    
    # Point differential
    df['POINT_DIFF'] = df['HOME_SCORE'] - df['AWAY_SCORE']
    
    # Total points
    df['TOTAL_POINTS'] = df['HOME_SCORE'] + df['AWAY_SCORE']
    
    return df

def create_betting_outcome_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create betting outcome features."""
    logger.info("Creating betting outcome features...")
    
    # Against the spread (ATS) outcome
    df['ATS_RESULT'] = np.where(
        df['POINT_DIFF'] > df['SPREAD'],
        1,
        np.where(
            df['POINT_DIFF'] < df['SPREAD'],
            -1,
            0
        )
    )
    
    # Over/Under outcome
    df['OVER_UNDER_RESULT'] = np.where(
        df['TOTAL_POINTS'] > df['TOTAL'],
        1,
        np.where(
            df['TOTAL_POINTS'] < df['TOTAL'],
            -1,
            0
        )
    )
    
    # Moneyline outcome
    df['MONEYLINE_RESULT'] = df['HOME_WIN']
    
    return df

def create_target_features(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create target features for model training."""
    logger.info("Starting target feature engineering...")
    
    # Get betting data
    df = data['validated_odds'].copy()
    
    # Create target variables
    logger.info("Creating target variables...")
    
    # Win/Loss target
    df['TARGET_WIN'] = (df['HOME_SCORE'] > df['AWAY_SCORE']).astype(int)
    
    # Spread target
    df['TARGET_SPREAD'] = df['HOME_SCORE'] - df['AWAY_SCORE']
    
    # Total points target
    df['TARGET_TOTAL'] = df['HOME_SCORE'] + df['AWAY_SCORE']
    
    # Over/Under target
    df['TARGET_OVER'] = (df['TARGET_TOTAL'] > df['FULL-GAME_BETMGM_TOTAL']).astype(int)
    
    logger.info(f"Created target features with shape {df.shape}")
    return df
