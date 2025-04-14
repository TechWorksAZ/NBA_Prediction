"""
Feature engineering script for NBA prediction model.
Creates features from core stats, advanced stats, and betting data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path("G:/My Drive/Projects/NBA_Prediction")
RAW_DIR = Path("C:/Projects/NBA_Prediction/data/raw")
PROCESSED_DIR = Path("C:/Projects/NBA_Prediction/data/processed")
FEATURES_DIR = PROCESSED_DIR / "features"

# Ensure directories exist
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

class FeatureEngineer:
    def __init__(self):
        self.core_stats: Optional[pd.DataFrame] = None
        self.advanced_stats: Optional[pd.DataFrame] = None
        self.betting_data: Optional[pd.DataFrame] = None
        
    def load_raw_data(self) -> None:
        """Load raw data from various sources."""
        try:
            # Load core stats
            core_path = RAW_DIR / "core" / "nba_team_stats.csv"
            if core_path.exists():
                self.core_stats = pd.read_csv(core_path)
                # Convert game_date to date and rename team column
                self.core_stats['date'] = pd.to_datetime(self.core_stats['game_date']).dt.date
                self.core_stats['team'] = self.core_stats['team_abbreviation']
                logger.info("Loaded core stats")
            else:
                logger.warning("Core stats file not found")
                
            # Load advanced stats
            advanced_path = RAW_DIR / "advanced" / "nba_advanced_stats.csv"
            if advanced_path.exists():
                self.advanced_stats = pd.read_csv(advanced_path)
                logger.info("Loaded advanced stats")
            else:
                logger.warning("Advanced stats file not found")
                
            # Load betting data
            betting_path = RAW_DIR / "betting" / "sbr_daily" / "sbr_odds_2025-04-11.csv"
            if betting_path.exists():
                self.betting_data = pd.read_csv(betting_path)
                # Convert date to datetime.date
                self.betting_data['date'] = pd.to_datetime(self.betting_data['date']).dt.date
                logger.info("Loaded betting data")
            else:
                logger.warning("Betting data file not found")
                
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise
            
    def create_rolling_features(self, df: pd.DataFrame, group_col: str, 
                              value_cols: List[str], windows: List[int],
                              date_col: Optional[str] = 'date') -> pd.DataFrame:
        """Create rolling average features for specified columns."""
        if date_col in df.columns:
            df = df.sort_values([date_col, group_col])
        else:
            df = df.sort_values(group_col)
            
        for window in windows:
            for col in value_cols:
                df[f'{col}_rolling_{window}'] = (
                    df.groupby(group_col)[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
        return df
        
    def create_opponent_adjusted_features(self, df: pd.DataFrame, 
                                        metrics: List[str]) -> pd.DataFrame:
        """Create opponent-adjusted features based on league averages."""
        # Calculate league averages for each metric
        league_avgs = df[metrics].mean()
        
        # Calculate opponent strength (how much better/worse than league average)
        opponent_strength = df.groupby('opponent')[metrics].mean() - league_avgs
        
        # Create opponent-adjusted features
        for metric in metrics:
            df[f'{metric}_opp_adj'] = (
                df[metric] - df['opponent'].map(opponent_strength[metric])
            )
            
        return df
        
    def process_betting_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process betting data to create features."""
        # Extract spread data for each sportsbook
        sportsbooks = ['betmgm', 'fanduel', 'caesars', 'bet365', 'draftkings', 'bet_rivers_az']
        
        for book in sportsbooks:
            # Extract full game spread for home team
            spread_col = f'full-game_{book}_spread_home'
            if spread_col in df.columns:
                df[f'{book}_spread'] = df[spread_col]
                
        # Calculate average spread across sportsbooks
        spread_cols = [f'{book}_spread' for book in sportsbooks if f'{book}_spread' in df.columns]
        if spread_cols:
            df['avg_spread'] = df[spread_cols].mean(axis=1)
            df['spread_std'] = df[spread_cols].std(axis=1)
            
        # Extract total points
        for book in sportsbooks:
            total_col = f'full-game_{book}_total'
            if total_col in df.columns:
                df[f'{book}_total'] = df[total_col]
                
        # Calculate average total points
        total_cols = [f'{book}_total' for book in sportsbooks if f'{book}_total' in df.columns]
        if total_cols:
            df['avg_total'] = df[total_cols].mean(axis=1)
            df['total_std'] = df[total_cols].std(axis=1)
            
        return df
        
    def engineer_features(self) -> pd.DataFrame:
        """Main function to engineer all features."""
        try:
            # Load raw data
            self.load_raw_data()
            
            # Initialize features DataFrame
            features_df = pd.DataFrame()
            
            # Process core stats if available
            if self.core_stats is not None:
                core_features = self.core_stats.copy()
                
                # Create rolling features for core stats
                core_metrics = ['pts', 'ast', 'reb', 'stl', 'blk']
                core_features = self.create_rolling_features(
                    core_features, 'team', core_metrics, [3, 5, 10]
                )
                
                # Add to features DataFrame
                features_df = pd.concat([features_df, core_features], axis=1)
                
            # Process advanced stats if available
            if self.advanced_stats is not None:
                advanced_features = self.advanced_stats.copy()
                
                # Create rolling features for advanced stats
                advanced_metrics = ['off_rating', 'def_rating', 'net_rating']
                advanced_features = self.create_rolling_features(
                    advanced_features, 'team', advanced_metrics, [3, 5, 10],
                    date_col=None  # No date column in advanced stats
                )
                
                # Add to features DataFrame
                features_df = pd.concat([features_df, advanced_features], axis=1)
                
            # Process betting data if available
            if self.betting_data is not None:
                betting_features = self.process_betting_data(self.betting_data.copy())
                
                # Add temporal features
                betting_features['date'] = pd.to_datetime(betting_features['date'])
                betting_features['day_of_week'] = betting_features['date'].dt.dayofweek
                betting_features['month'] = betting_features['date'].dt.month
                betting_features['is_weekend'] = betting_features['day_of_week'].isin([5, 6]).astype(int)
                
                # Add to features DataFrame
                features_df = pd.concat([features_df, betting_features], axis=1)
                
            return features_df
            
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {str(e)}")
            raise
            
    def save_features(self, df: pd.DataFrame) -> None:
        """Save engineered features to disk."""
        try:
            output_path = FEATURES_DIR / "engineered_features.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved engineered features to {output_path}")
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise

def main():
    """Main function to run the feature engineering pipeline."""
    try:
        logger.info("Starting feature engineering pipeline...")
        engineer = FeatureEngineer()
        features_df = engineer.engineer_features()
        engineer.save_features(features_df)
        logger.info("Feature engineering pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 