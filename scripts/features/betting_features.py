"""
Betting Feature Engineering Module

This module creates features from betting data including:
- Current lines and odds
- Betting percentages
- Public betting trends
- Sharp money indicators

Author: NBA Prediction Team
Date: 2024
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from scripts.features.column_mappings import VALIDATED_ODDS_COLS, get_col

logger = logging.getLogger(__name__)

def create_line_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from current lines."""
    logger.info("Creating line features...")
    
    # Calculate spread and total features
    df['SPREAD'] = df['FULL-GAME_BETMGM_SPREAD_HOME']
    df['TOTAL'] = df['FULL-GAME_BETMGM_TOTAL']
    
    # Calculate spread and total percentages
    df['SPREAD_PCT'] = df['SPREAD'] / df['SPREAD'].abs()
    df['TOTAL_PCT'] = df['TOTAL'] / df['TOTAL'].mean()
    
    return df

def create_betting_percentage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from betting percentages."""
    logger.info("Creating betting percentage features...")
    
    # Calculate betting percentage differences
    df['SPREAD_BET_PCT_DIFF'] = df['FULL-GAME_BETMGM_SPREAD_HOME_ODDS'] - df['FULL-GAME_BETMGM_SPREAD_AWAY_ODDS']
    df['MONEYLINE_BET_PCT_DIFF'] = df['FULL-GAME_BETMGM_MONEYLINE_HOME'] - df['FULL-GAME_BETMGM_MONEYLINE_AWAY']
    df['TOTAL_BET_PCT_DIFF'] = df['FULL-GAME_BETMGM_TOTAL_OVER_ODDS'] - df['FULL-GAME_BETMGM_TOTAL_UNDER_ODDS']
    
    # Calculate sharp money indicators based on betting percentages only
    df['SHARP_MONEY_SPREAD'] = np.where(
        df['FULL-GAME_BETMGM_SPREAD_HOME_ODDS'] > 0.7,
        1,
        np.where(
            df['FULL-GAME_BETMGM_SPREAD_AWAY_ODDS'] > 0.7,
            -1,
            0
        )
    )
    
    df['SHARP_MONEY_TOTAL'] = np.where(
        df['FULL-GAME_BETMGM_TOTAL_OVER_ODDS'] > 0.7,
        1,
        np.where(
            df['FULL-GAME_BETMGM_TOTAL_UNDER_ODDS'] > 0.7,
            -1,
            0
        )
    )
    
    return df

def create_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from odds data."""
    logger.info("Creating odds features...")
    
    # Calculate implied probabilities
    df['HOME_IMPLIED_PROB'] = 1 / (1 + df['FULL-GAME_BETMGM_MONEYLINE_HOME'])
    df['AWAY_IMPLIED_PROB'] = 1 / (1 + df['FULL-GAME_BETMGM_MONEYLINE_AWAY'])
    
    # Calculate vig
    df['VIG'] = df['HOME_IMPLIED_PROB'] + df['AWAY_IMPLIED_PROB'] - 1
    
    # Calculate fair odds
    df['FAIR_HOME_ODDS'] = df['FULL-GAME_BETMGM_MONEYLINE_HOME'] * (1 - df['VIG']/2)
    df['FAIR_AWAY_ODDS'] = df['FULL-GAME_BETMGM_MONEYLINE_AWAY'] * (1 - df['VIG']/2)
    
    return df

def create_betting_features(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create betting-related features."""
    logger.info("Starting betting feature engineering...")
    
    # Get odds data
    odds_df = data['validated_odds'].copy()
    
    # Create line features
    logger.info("Creating line features...")
    odds_df = create_line_features(odds_df)
    
    # Create betting percentage features
    logger.info("Creating betting percentage features...")
    odds_df = create_betting_percentage_features(odds_df)
    
    # Create odds features
    logger.info("Creating odds features...")
    odds_df = create_odds_features(odds_df)
    
    logger.info(f"Created betting features with shape {odds_df.shape}")
    return odds_df
