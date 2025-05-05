"""
NBA Feature Engineering Pipeline

This script processes the cleaned data and creates features for the prediction models.
Features include:
- Rolling averages (3, 5, 10 games)
- Opponent-adjusted statistics
- Matchup-specific features
- Team and player efficiency metrics
- Game context features
- Advanced metrics for improved accuracy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
from geopy.distance import geodesic
import traceback
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import seaborn as sns

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

class FeatureEngineer:
    def __init__(self, base_dir: str = "C:/projects/NBA_prediction"):
        """Initialize the feature engineering class."""
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "data" / "processed"
        self.features_dir = self.base_dir / "data" / "processed" / "features"
        self.raw_dir = self.base_dir / "data" / "raw"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize dataframes
        self.games_df = pd.DataFrame()
        self.odds_df = pd.DataFrame()
        self.advanced_player_df = pd.DataFrame()
        self.advanced_team_df = pd.DataFrame()
        self.tracking_player_df = pd.DataFrame()
        self.tracking_team_df = pd.DataFrame()
        self.matchups_player_df = pd.DataFrame()
        self.core_player_df = pd.DataFrame()
        
        # Initialize feature dataframes
        self.team_features_df = pd.DataFrame()
        self.player_features_df = pd.DataFrame()
        self.game_features_df = pd.DataFrame()
        self.betting_features_df = pd.DataFrame()
        self.shot_metrics_df = pd.DataFrame()
        
        # Load team locations for travel distance calculation
        self.team_locations = self._load_team_locations()
        
        # Load and process data
        self.load_data()
        
        # Load player features
        self.load_player_features()
        
        # Handle date formats
        self._standardize_date_formats()
        
        self.logger.info("FeatureEngineer initialized successfully")

    def _standardize_date_formats(self):
        """Standardize date formats across all dataframes."""
        date_cols = ['GAME_DATE', 'DATE']
        for df_name in ['games_df', 'odds_df']:
            df = getattr(self, df_name)
            if not df.empty:
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert all game_id columns to string type
        for df_name in ['games_df', 'odds_df']:
            df = getattr(self, df_name)
            if not df.empty and 'GAME_ID' in df.columns:
                df['GAME_ID'] = df['GAME_ID'].astype(str)

    def load_data(self) -> None:
        """Load and validate all required data files."""
        self.logger.info("Loading processed data files...")
        
        try:
            # Load validated games data
            games_path = self.processed_dir / "validated_games.csv"
            if games_path.exists():
                self.games_df = pd.read_csv(games_path)
                self.games_df = self.games_df[~self.games_df['GAME_ID'].astype(str).str.contains('003|001')]
                self.games_df = self._standardize_columns(self.games_df)
                self.logger.info(f"Loaded {len(self.games_df)} validated games")
            else:
                self.logger.error("Games data file not found")
                raise ValueError("Games data file not found")
            
            # Load validated odds data
            odds_path = self.processed_dir / "validated_odds.csv"
            if odds_path.exists():
                self.odds_df = pd.read_csv(odds_path)
                if 'game_id' in self.odds_df.columns:
                    self.odds_df = self.odds_df[~self.odds_df['game_id'].astype(str).str.contains('003|001')]
                self.odds_df = self._standardize_columns(self.odds_df)
                self.logger.info(f"Loaded {len(self.odds_df)} validated odds records")
            else:
                self.logger.error("Odds data file not found")
                raise ValueError("Odds data file not found")
            
            # Load advanced player data
            adv_player_path = self.processed_dir / "advanced_p_data.csv"
            if adv_player_path.exists():
                self.advanced_player_df = pd.read_csv(adv_player_path)
                self.advanced_player_df = self._standardize_columns(self.advanced_player_df)
                self.logger.info(f"Loaded {len(self.advanced_player_df)} advanced player records")
            else:
                self.logger.warning("Advanced player data file not found")
                self.advanced_player_df = pd.DataFrame()
            
            # Load tracking player data
            tracking_player_path = self.processed_dir / "tracking_p_data.csv"
            if tracking_player_path.exists():
                self.tracking_player_df = pd.read_csv(tracking_player_path)
                self.tracking_player_df = self._standardize_columns(self.tracking_player_df)
                self.logger.info(f"Loaded {len(self.tracking_player_df)} tracking player records")
            else:
                self.logger.warning("Tracking player data file not found")
                self.tracking_player_df = pd.DataFrame()
            
            # Load advanced team data
            adv_team_path = self.processed_dir / "advanced_t_data.csv"
            if adv_team_path.exists():
                self.advanced_team_df = pd.read_csv(adv_team_path)
                self.advanced_team_df = self._standardize_columns(self.advanced_team_df)
                self.logger.info(f"Loaded {len(self.advanced_team_df)} advanced team records")
            else:
                self.logger.warning("Advanced team data file not found")
                self.advanced_team_df = pd.DataFrame()
            
            # Load tracking team data
            tracking_team_path = self.processed_dir / "tracking_t_data.csv"
            if tracking_team_path.exists():
                self.tracking_team_df = pd.read_csv(tracking_team_path)
                self.tracking_team_df = self._standardize_columns(self.tracking_team_df)
                self.logger.info(f"Loaded {len(self.tracking_team_df)} tracking team records")
            else:
                self.logger.warning("Tracking team data file not found")
                self.tracking_team_df = pd.DataFrame()
            
            # Load matchups player data
            matchups_player_path = self.processed_dir / "matchups_p_data.csv"
            if matchups_player_path.exists():
                self.matchups_player_df = pd.read_csv(matchups_player_path)
                self.matchups_player_df = self._standardize_columns(self.matchups_player_df)
                self.logger.info(f"Loaded {len(self.matchups_player_df)} matchups player records")
            else:
                self.logger.warning("Matchups player data file not found")
                self.matchups_player_df = pd.DataFrame()
            
            # Load core player data
            core_player_path = self.processed_dir / "core_p_data.csv"
            if core_player_path.exists():
                self.core_player_df = pd.read_csv(core_player_path)
                self.core_player_df = self._standardize_columns(self.core_player_df)
                self.logger.info(f"Loaded {len(self.core_player_df)} core player records")
            else:
                self.logger.warning("Core player data file not found")
                self.core_player_df = pd.DataFrame()
            
            self.logger.info("Data loading completed")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _load_team_locations(self) -> Dict[str, Tuple[float, float]]:
        """Load team locations for travel distance calculation."""
        # This is a simplified version - in production, you'd want to load actual coordinates
        return {
            'ATL': (33.7573, -84.3963),  # Atlanta
            'BOS': (42.3662, -71.0621),  # Boston
            'BKN': (40.6826, -73.9754),  # Brooklyn
            'CHA': (35.2251, -80.8392),  # Charlotte
            'CHI': (41.8806, -87.6742),  # Chicago
            'CLE': (41.4965, -81.6882),  # Cleveland
            'DAL': (32.7903, -96.8103),  # Dallas
            'DEN': (39.7487, -105.0076), # Denver
            'DET': (42.3411, -83.0558),  # Detroit
            'GSW': (37.7680, -122.3875), # Golden State
            'HOU': (29.7508, -95.3621),  # Houston
            'IND': (39.7639, -86.1555),  # Indiana
            'LAC': (34.0430, -118.2673), # LA Clippers
            'LAL': (34.0430, -118.2673), # LA Lakers
            'MEM': (35.1380, -90.0506),  # Memphis
            'MIA': (25.7814, -80.1866),  # Miami
            'MIL': (43.0436, -87.9169),  # Milwaukee
            'MIN': (44.9795, -93.2761),  # Minnesota
            'NOP': (29.9490, -90.0818),  # New Orleans
            'NYK': (40.7505, -73.9934),  # New York
            'OKC': (35.4634, -97.5151),  # Oklahoma City
            'ORL': (28.5392, -81.3839),  # Orlando
            'PHI': (39.9012, -75.1719),  # Philadelphia
            'PHX': (33.4457, -112.0712), # Phoenix
            'POR': (45.5316, -122.6668), # Portland
            'SAC': (38.5802, -121.4996), # Sacramento
            'SAS': (29.4269, -98.4375),  # San Antonio
            'TOR': (43.6435, -79.3791),  # Toronto
            'UTA': (40.7683, -111.9011), # Utah
            'WAS': (38.8981, -77.0209)   # Washington
        }
    
    def calculate_rolling_stats(self, df: pd.DataFrame, group_cols: List[str], 
                              stat_cols: List[str], windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """Calculate rolling statistics for specified columns."""
        try:
            # Create a copy to avoid fragmentation
            df = df.copy()
            
            # Sort by date and game ID
            if 'game_date' in df.columns:
                df = df.sort_values(['game_date', 'game_id'])
            
            # Collect all new columns in a list
            new_columns = []
            
            for window in windows:
                for col in stat_cols:
                    if col in df.columns:
                        # Calculate rolling mean
                        new_col = f'{col}_rolling_{window}'
                        df[new_col] = df.groupby(group_cols)[col].transform(
                            lambda x: x.rolling(window, min_periods=1).mean()
                        )
                        
                        # Calculate rolling std
                        std_col = f'{col}_rolling_{window}_std'
                        df[std_col] = df.groupby(group_cols)[col].transform(
                            lambda x: x.rolling(window, min_periods=1).std()
                        )
                    else:
                        logger.warning(f"Column {col} not found in dataframe for rolling stats")
            
            logger.info("Successfully calculated rolling statistics")
        except Exception as e:
            logger.error(f"Error calculating rolling stats: {str(e)}")
        
        return df
    
    def calculate_efficiency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team efficiency metrics from base stats.
        
        Args:
            df (pd.DataFrame): Input DataFrame with team stats
            
        Returns:
            pd.DataFrame: DataFrame with added efficiency metrics
        """
        try:
            # Check required columns
            required_cols = ['PTS', 'FGA', 'FTA', 'OREB', 'DREB', 'TOV', 'POSS']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.warning(f"Missing columns for efficiency metrics: {missing_cols}")
                return df
                
            # Calculate Four Factors
            if all(col in df.columns for col in ['FGM', 'FGA']):
                df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
                
            if all(col in df.columns for col in ['TOV', 'POSS']):
                df['TOV_PCT'] = df['TOV'] / df['POSS'] * 100
                
            if all(col in df.columns for col in ['OREB', 'DREB']):
                opp_dreb = df.groupby('GAME_ID')['DREB'].transform(lambda x: x.sum() - x)
                df['OREB_PCT'] = df['OREB'] / (df['OREB'] + opp_dreb) * 100
                
            if all(col in df.columns for col in ['FTA', 'FGA']):
                df['FT_RATE'] = df['FTA'] / df['FGA']
            
            # Calculate additional efficiency metrics
            if all(col in df.columns for col in ['AST', 'FGM']):
                df['AST_PCT'] = df['AST'] / df['FGM'] * 100
                
            if all(col in df.columns for col in ['STL', 'POSS']):
                df['STL_PCT'] = df['STL'] / df['POSS'] * 100
                
            if all(col in df.columns for col in ['BLK', 'OPP_FGA']):
                df['BLK_PCT'] = df['BLK'] / df['OPP_FGA'] * 100
                
            # Calculate overall offensive and defensive ratings
            if 'POSS' in df.columns:
                df['OFFENSIVE_RATING'] = df['PTS'] / df['POSS'] * 100
                opp_pts = df.groupby('GAME_ID')['PTS'].transform(lambda x: x.sum() - x)
                df['DEFENSIVE_RATING'] = opp_pts / df['POSS'] * 100
        
            return df
            
        except Exception as e:
            self.logger.error(f"Error in calculate_efficiency_metrics: {str(e)}")
            self.logger.error(traceback.format_exc())
            return df
    
    def calculate_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced metrics based on available data."""
        try:
            # Create a copy to avoid fragmentation
            df = df.copy()
            
            # Calculate advanced metrics based on available columns
            if 'PTS' in df.columns:
                # Calculate scoring consistency (standard deviation of points)
                df['SCORING_CONSISTENCY'] = df.groupby('PLAYER_ID')['PTS'].transform('std')
                
                # Calculate scoring trend (difference between recent and overall average)
                df['RECENT_PPG'] = df.groupby('PLAYER_ID')['PTS'].transform(
                    lambda x: x.rolling(5, min_periods=1).mean()
                )
                df['OVERALL_PPG'] = df.groupby('PLAYER_ID')['PTS'].transform('mean')
                df['SCORING_TREND'] = df['RECENT_PPG'] - df['OVERALL_PPG']
            
            if 'PLUS_MINUS' in df.columns:
                # Calculate impact consistency (standard deviation of plus/minus)
                df['IMPACT_CONSISTENCY'] = df.groupby('PLAYER_ID')['PLUS_MINUS'].transform('std')
                
                # Calculate impact trend
                df['RECENT_IMPACT'] = df.groupby('PLAYER_ID')['PLUS_MINUS'].transform(
                    lambda x: x.rolling(5, min_periods=1).mean()
                )
                df['OVERALL_IMPACT'] = df.groupby('PLAYER_ID')['PLUS_MINUS'].transform('mean')
                df['IMPACT_TREND'] = df['RECENT_IMPACT'] - df['OVERALL_IMPACT']
            
            logger.info("Successfully calculated advanced metrics")
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {str(e)}")
        
        return df
    
    def calculate_team_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team ratings based on available statistics."""
        logger.info("Calculating team ratings...")
        
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Validate required columns
            required_cols = ['team_score', 'opponent_score', 'team_id', 'opponent_team_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns for team ratings: {missing_cols}")
                return df
            
            # Calculate basic offensive and defensive ratings
            df['offensive_rating'] = df['team_score']
            df['defensive_rating'] = df['opponent_score']
            
            # Calculate net ratings
            df['net_rating'] = df['offensive_rating'] - df['defensive_rating']
            
            # Calculate rolling averages for ratings
            for window in [3, 5, 10]:
                # Team rolling ratings
                df[f'off_rating_{window}g'] = df.groupby('team_id')['offensive_rating'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'def_rating_{window}g'] = df.groupby('team_id')['defensive_rating'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'net_rating_{window}g'] = df.groupby('team_id')['net_rating'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
            # Calculate rating differentials
            df['off_rating_diff'] = df.groupby('team_id')['offensive_rating'].transform(
                lambda x: x - x.shift(1)
            )
            df['def_rating_diff'] = df.groupby('team_id')['defensive_rating'].transform(
                lambda x: x - x.shift(1)
            )
            df['net_rating_diff'] = df.groupby('team_id')['net_rating'].transform(
                lambda x: x - x.shift(1)
            )
            
            # Calculate rating trends
            for window in [3, 5]:
                # Team trends
                df[f'off_trend_{window}g'] = df.groupby('team_id')['offensive_rating'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean() - x.rolling(window*2, min_periods=1).mean()
                )
                df[f'def_trend_{window}g'] = df.groupby('team_id')['defensive_rating'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean() - x.rolling(window*2, min_periods=1).mean()
                )
            
            logger.info("Successfully calculated team ratings")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating team ratings: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return df
    
    def calculate_pace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pace (possessions per 48 minutes)."""
        logger.info("Calculating pace...")
        
        # Create a copy to avoid fragmentation
        df = df.copy()
        
        try:
            # Check for required columns
            required_cols = ['field_goals_attempted', 'offensive_rebounds', 'turnovers', 'free_throws_attempted', 'minutes']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Cannot calculate pace - missing required columns: {missing_cols}")
                # Try to estimate pace using available columns
                if 'team_score' in df.columns and 'opponent_team_score' in df.columns:
                    logger.info("Estimating pace using team scores")
                    df['pace'] = (df['team_score'] + df['opponent_team_score']) / 2
                return df
            
            # Calculate possessions
            df['possessions'] = (
                df['field_goals_attempted'] - 
                df['offensive_rebounds'] + 
                df['turnovers'] + 
                0.44 * df['free_throws_attempted']
            )
            
            # Calculate pace (possessions per 48 minutes)
            df['pace'] = (df['possessions'] / df['minutes']) * 48
            logger.info("Successfully calculated pace")
        except Exception as e:
            logger.error(f"Error calculating pace: {str(e)}")
        
        return df
    
    def calculate_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate matchup-specific features."""
        logger.info("Calculating matchup features...")
        
        try:
            # Calculate point differential for matchups
            matchup_stats = df.groupby(['TEAM_ID', 'OPPONENT_TEAM_ID'], group_keys=False).agg({
                'TEAM_SCORE': ['mean', 'std'],
                'OPPONENT_SCORE': ['mean', 'std']
            }).reset_index()
            
            # Flatten column names
            matchup_stats.columns = ['TEAM_ID', 'OPPONENT_TEAM_ID', 
                                   'MATCHUP_AVG_TEAM_SCORE', 'MATCHUP_STD_TEAM_SCORE',
                                   'MATCHUP_AVG_OPP_SCORE', 'MATCHUP_STD_OPP_SCORE']
            
            # Calculate matchup point differential
            matchup_stats['MATCHUP_POINT_DIFF'] = (
                matchup_stats['MATCHUP_AVG_TEAM_SCORE'] - matchup_stats['MATCHUP_AVG_OPP_SCORE']
            )
            
            # Merge back to main dataframe
            df = df.merge(
                matchup_stats,
                on=['TEAM_ID', 'OPPONENT_TEAM_ID'],
                how='left'
            )
            
            # Add player matchup data if available
            if not self.matchups_player_df.empty:
                logger.info("Adding player matchup data...")
                # Aggregate player matchup stats by team
                player_matchup_stats = self.matchups_player_df.groupby(['DEF_TEAM_ID', 'OFF_TEAM_ID']).agg({
                    'MATCHUP_MIN': 'sum',
                    'PARTIAL_POSS': 'sum',
                    'PLAYER_PTS': 'sum',
                    'TEAM_PTS': 'sum',
                    'MATCHUP_AST': 'sum',
                    'MATCHUP_TOV': 'sum',
                    'MATCHUP_BLK': 'sum',
                    'MATCHUP_FG_PCT': 'mean',
                    'MATCHUP_FG3_PCT': 'mean'
                }).reset_index()
                
                # Rename columns to match our format
                player_matchup_stats = player_matchup_stats.rename(columns={
                    'DEF_TEAM_ID': 'TEAM_ID',
                    'OFF_TEAM_ID': 'OPPONENT_TEAM_ID'
                })
                
                # Merge with main dataframe
                df = df.merge(
                    player_matchup_stats,
                    on=['TEAM_ID', 'OPPONENT_TEAM_ID'],
                    how='left',
                    suffixes=('', '_matchup')
                )
            
            # Add admin matchup data if available
            if not self.matchups_admin_df.empty:
                logger.info("Adding admin matchup data...")
                # Aggregate admin matchup stats by team
                admin_matchup_stats = self.matchups_admin_df.groupby(['TEAM_ID', 'OPPONENT_TEAM_ID']).agg({
                    'MATCHUP_MIN': 'sum',
                    'PARTIAL_POSS': 'sum',
                    'PLAYER_PTS': 'sum',
                    'TEAM_PTS': 'sum',
                    'MATCHUP_AST': 'sum',
                    'MATCHUP_TOV': 'sum',
                    'MATCHUP_BLK': 'sum',
                    'MATCHUP_FG_PCT': 'mean',
                    'MATCHUP_FG3_PCT': 'mean'
                }).reset_index()
                
                # Merge with main dataframe
                df = df.merge(
                    admin_matchup_stats,
                    on=['TEAM_ID', 'OPPONENT_TEAM_ID'],
                    how='left',
                    suffixes=('', '_admin')
                )
            
            # Calculate style compatibility
            if all(col in df.columns for col in ['PACE', 'OPPONENT_PACE', 'OFF_RATING', 'DEF_RATING']):
                # Pace compatibility (closer to 0 means more compatible)
                df['PACE_COMPATIBILITY'] = abs(df['PACE'] - df['OPPONENT_PACE'])
                
                # Offensive vs defensive rating compatibility
                df['OFF_DEF_COMPATIBILITY'] = abs(df['OFF_RATING'] - df['DEF_RATING'])
            else:
                logger.warning("Missing required columns for style compatibility. Required: " +
                             "['PACE', 'OPPONENT_PACE', 'OFF_RATING', 'DEF_RATING']")
            
            logger.info("Successfully calculated matchup features")
        except Exception as e:
            logger.error(f"Error calculating matchup features: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        return df
    
    def calculate_game_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate game context features like rest, travel, etc."""
        logger.info("Calculating game context features...")
        
        try:
            # Convert game_date to datetime if it's not already
            if 'GAME_DATE' in df.columns:
                try:
                    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
                    logger.info("Converted GAME_DATE to datetime")
                except Exception as e:
                    logger.warning(f"Could not convert GAME_DATE to datetime: {str(e)}")
            
            # Back-to-back games
            if 'GAME_DATE' in df.columns and isinstance(df['GAME_DATE'].iloc[0], pd.Timestamp):
                df['BACK_TO_BACK'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days == 1
                logger.info("Calculated back-to-back games")
            
            # Travel distance
            if all(col in df.columns for col in ['TEAM_ABBR', 'OPPONENT_TEAM_ABBR']):
                df['TRAVEL_DISTANCE'] = df.apply(
                    lambda x: geodesic(
                        self.team_locations.get(x['TEAM_ABBR'], (0, 0)),
                        self.team_locations.get(x['OPPONENT_TEAM_ABBR'], (0, 0))
                    ).miles if x['TEAM_ABBR'] in self.team_locations and x['OPPONENT_TEAM_ABBR'] in self.team_locations else 0,
                    axis=1
                )
                logger.info("Calculated travel distances")
            
            # Rest days
            if 'GAME_DATE' in df.columns and isinstance(df['GAME_DATE'].iloc[0], pd.Timestamp):
                df['REST_DAYS'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
                df['OPPONENT_REST_DAYS'] = df.groupby('opponent_team_id')['GAME_DATE'].diff().dt.days
                df['REST_ADVANTAGE'] = df['REST_DAYS'] - df['OPPONENT_REST_DAYS']
                logger.info("Calculated rest days and advantage")
            
            # Team momentum (last 5 games win percentage)
            if 'WIN' in df.columns:
                df['MOMENTUM'] = df.groupby('TEAM_ID')['WIN'].transform(
                    lambda x: x.rolling(5, min_periods=1).mean()
                )
                logger.info("Calculated team momentum")
            
            return df
        except Exception as e:
            logger.error(f"Error calculating game context features: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return df
    
    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features like season progress, game timing, etc."""
        logger.info("Calculating time-based features...")
        
        try:
            df = df.copy()
            
            if 'GAME_DATE' in df.columns:
                try:
                    # Ensure game_date is datetime
                    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
                    
                    # Extract basic time components
                    df['MONTH'] = df['GAME_DATE'].dt.month
                    df['DAY_OF_WEEK'] = df['GAME_DATE'].dt.dayofweek
                    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)
                    
                    # Calculate days since season start for each season
                    df['SEASON'] = df['GAME_DATE'].dt.year.where(
                        df['GAME_DATE'].dt.month > 8,
                        df['GAME_DATE'].dt.year - 1
                    )
                    
                    # Calculate season start dates
                    season_starts = df.groupby('SEASON')['GAME_DATE'].transform('min')
                    df['DAYS_INTO_SEASON'] = (df['GAME_DATE'] - season_starts).dt.days
                    
                    # Calculate season progress (0-1 scale)
                    season_ends = df.groupby('SEASON')['GAME_DATE'].transform('max')
                    season_lengths = (season_ends - season_starts).dt.days
                    df['SEASON_PROGRESS'] = df['DAYS_INTO_SEASON'] / season_lengths
                    
                    # Days since last game (team-specific)
                    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])
                    df['DAYS_SINCE_LAST_GAME'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
                    
                    # Game density (games in last 7/14 days)
                    for days in [7, 14]:
                        df[f'GAMES_LAST_{days}_DAYS'] = df.groupby('TEAM_ID').apply(
                            lambda group: group.apply(
                                lambda row: len(group[
                                    (group['GAME_DATE'] <= row['GAME_DATE']) & 
                                    (group['GAME_DATE'] > row['GAME_DATE'] - pd.Timedelta(days=days))
                                ])
                            )
                        ).reset_index(level=0, drop=True)
                    
                    logger.info("Successfully calculated time-based features")
                except Exception as e:
                    logger.error(f"Error calculating time features: {str(e)}")
                    # Add basic time features even if advanced calculations fail
                    df['MONTH'] = df['GAME_DATE'].dt.month
                    df['DAY_OF_WEEK'] = df['GAME_DATE'].dt.dayofweek
                    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)
            else:
                logger.warning("Cannot calculate time features - missing GAME_DATE column")
            
            return df
        except Exception as e:
            logger.error(f"Error calculating time features: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return df
    
    def calculate_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate streak-based features for teams."""
        logger.info("Calculating streak features...")
        
        try:
            # Sort by date to calculate streaks
            df = df.sort_values(['TEAM_ID', 'GAME_DATE'])
            
            # Calculate win/loss
            df['WIN'] = (df['TEAM_SCORE'] > df['OPPONENT_SCORE']).astype(int)
            
            # Calculate point differential
            df['POINT_DIFF'] = df['TEAM_SCORE'] - df['OPPONENT_SCORE']
            
            # Initialize streak features
            # Win/Loss streaks
            df['WIN_STREAK'] = df.groupby('TEAM_ID')['WIN'].transform(
                lambda x: x.rolling(window=10, min_periods=1).sum()
            )
            
            # Point differential streaks
            df['POINT_DIFF_STREAK'] = df.groupby('TEAM_ID')['POINT_DIFF'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            
            # Last 3, 5, 10 games performance
            for window in [3, 5, 10]:
                # Win percentage
                df[f'LAST_{window}_WIN_PCT'] = df.groupby('TEAM_ID')['WIN'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Average point differential
                df[f'LAST_{window}_POINT_DIFF'] = df.groupby('TEAM_ID')['POINT_DIFF'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            
            logger.info("Successfully calculated streak features")
            
        except Exception as e:
            logger.error(f"Error calculating streak features: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        return df
    
    def calculate_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate head-to-head features."""
        try:
            if 'team_score' not in df.columns or 'opponent_team_score' not in df.columns:
                logger.warning("Missing required columns for head-to-head features")
                return df
            
            # Sort by date for rolling calculations
            df = df.sort_values(['team_id', 'opponent_team_id', 'game_date'])
            
            # Calculate head-to-head average combined score
            h2h_scores = df.groupby(['team_id', 'opponent_team_id']).agg({
                'team_score': 'mean',
                'opponent_team_score': 'mean'
            }).reset_index()
            
            h2h_scores['h2h_avg_combined_score'] = h2h_scores['team_score'] + h2h_scores['opponent_team_score']
            
            # Merge back to main dataframe
            df = df.merge(
                h2h_scores[['team_id', 'opponent_team_id', 'h2h_avg_combined_score']],
                on=['team_id', 'opponent_team_id'],
                how='left'
            )
            
            logger.info("Successfully calculated head-to-head features")
        except Exception as e:
            logger.error(f"Error calculating head-to-head features: {str(e)}")
        return df
    
    def calculate_betting_features(self, df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate betting-related features."""
        logger.info("Calculating betting features...")
        
        try:
            # Ensure game_id is string type in both dataframes
            df['GAME_ID'] = df['GAME_ID'].astype(str)
            odds_df['GAME_ID'] = odds_df['GAME_ID'].astype(str)
            
            # Merge odds data
            df = df.merge(
                odds_df[['GAME_ID', 'HOME_SPREAD', 'AWAY_SPREAD', 'TOTAL', 
                        'HOME_MONEYLINE', 'AWAY_MONEYLINE', 'HOME_SCORE_H1',
                        'AWAY_SCORE_H1', 'HOME_SPREAD_H1', 'AWAY_SPREAD_H1',
                        'TOTAL_H1']],
                on='GAME_ID',
                how='left'
            )
            
            # Calculate implied probabilities from moneyline odds
            def moneyline_to_prob(moneyline):
                if pd.isna(moneyline):
                    return np.nan
                if moneyline > 0:
                    return 100 / (moneyline + 100)
                else:
                    return (-moneyline) / (-moneyline + 100)
            
            # Full game probabilities
            df['IMPLIED_WIN_PROB'] = np.where(
                df['IS_HOME'],
                df['HOME_MONEYLINE'].apply(moneyline_to_prob),
                df['AWAY_MONEYLINE'].apply(moneyline_to_prob)
            )
            
            # Calculate spread for each team (positive = underdog, negative = favorite)
            df['SPREAD'] = np.where(
                df['IS_HOME'],
                df['HOME_SPREAD'],
                df['AWAY_SPREAD']
            )
            
            # Calculate if team covered the spread
            df['COVERED_SPREAD'] = (df['TEAM_SCORE'] - df['OPPONENT_SCORE']) > df['SPREAD']
            
            # Calculate over/under result
            df['TOTAL_POINTS'] = df['TEAM_SCORE'] + df['OPPONENT_SCORE']
            df['OVER'] = df['TOTAL_POINTS'] > df['TOTAL']
            
            # First half features
            if 'HOME_SCORE_H1' in df.columns and 'AWAY_SCORE_H1' in df.columns:
                # First half spread
                df['SPREAD_H1'] = np.where(
                    df['IS_HOME'],
                    df['HOME_SPREAD_H1'],
                    df['AWAY_SPREAD_H1']
                )
                
                # First half point differential
                df['POINT_DIFF_H1'] = np.where(
                    df['IS_HOME'],
                    df['HOME_SCORE_H1'] - df['AWAY_SCORE_H1'],
                    df['AWAY_SCORE_H1'] - df['HOME_SCORE_H1']
                )
                
                # First half total points
                df['TOTAL_POINTS_H1'] = df['HOME_SCORE_H1'] + df['AWAY_SCORE_H1']
                
                # First half over/under result
                df['OVER_H1'] = df['TOTAL_POINTS_H1'] > df['TOTAL_H1']
                
                # First half spread cover
                df['COVERED_SPREAD_H1'] = df['POINT_DIFF_H1'] > df['SPREAD_H1']
            
            # Calculate rolling betting performance
            for window in [3, 5, 10]:
                # Full game rolling metrics
                df[f'COVER_RATE_{window}G'] = df.groupby('TEAM_ID')['COVERED_SPREAD'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'OVER_RATE_{window}G'] = df.groupby('TEAM_ID')['OVER'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
                # First half rolling metrics
                if 'COVERED_SPREAD_H1' in df.columns:
                    df[f'COVER_RATE_H1_{window}G'] = df.groupby('TEAM_ID')['COVERED_SPREAD_H1'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    df[f'OVER_RATE_H1_{window}G'] = df.groupby('TEAM_ID')['OVER_H1'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
            
            # Calculate market efficiency metrics
            df['PREDICTION_ERROR'] = df['IMPLIED_WIN_PROB'] - df['WIN']
            df['SPREAD_ERROR'] = df['SPREAD'] - (df['TEAM_SCORE'] - df['OPPONENT_SCORE'])
            df['TOTAL_ERROR'] = df['TOTAL'] - df['TOTAL_POINTS']
            
            # Calculate betting value metrics
            df['SPREAD_VALUE'] = df['SPREAD'] - df['POINT_DIFFERENTIAL'].rolling(10, min_periods=1).mean()
            df['TOTAL_VALUE'] = df['TOTAL'] - df['TOTAL_POINTS'].rolling(10, min_periods=1).mean()
            
            logger.info("Successfully calculated betting features")
        except Exception as e:
            logger.error(f"Error calculating betting features: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and types in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with standardized columns
        """
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Convert all column names to uppercase
            df.columns = [col.upper() for col in df.columns]
            
            # Standard column mappings
            column_mappings = {
                'PLAYER_ID': 'PERSON_ID',
                'PLAYER1_ID': 'PERSON_ID',
                'PLAYER2_ID': 'DEFENDER_ID',
                'AWAY_TEAM_ID': 'OPPONENT_TEAM_ID',
                'DATE': 'GAME_DATE'
            }
            
            # Apply mappings if columns exist
            for old_col, new_col in column_mappings.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Ensure GAME_ID is string type
            if 'GAME_ID' in df.columns:
                df['GAME_ID'] = df['GAME_ID'].astype(str)
            
            # Convert date columns to datetime if they exist
            date_cols = ['GAME_DATE', 'DATE']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')
            
            # Convert numeric columns to appropriate types
            numeric_cols = ['MINUTES', 'POINTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'OFFENSIVE_RATING', 'DEFENSIVE_RATING', 'NET_RATING']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in _standardize_columns: {str(e)}")
            self.logger.error(traceback.format_exc())
            return df  # Return original DataFrame if error occurs

    def calculate_pace_adjusted_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pace-adjusted statistics for team features.
        
        Args:
            df: DataFrame containing team features
            
        Returns:
            DataFrame with pace-adjusted statistics added
        """
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Calculate possessions if not present
            if 'possessions' not in df.columns:
                required_cols = ['field_goals_attempted', 'offensive_rebounds', 'turnovers', 'free_throws_attempted']
                if all(col in df.columns for col in required_cols):
                    df['possessions'] = (
                        df['field_goals_attempted'] - 
                        df['offensive_rebounds'] + 
                        df['turnovers'] + 
                        0.44 * df['free_throws_attempted']
                    )
                    logger.info("Calculated possessions from available statistics")
                else:
                    # Estimate possessions from points if detailed stats not available
                    if 'team_score' in df.columns and 'opponent_score' in df.columns:
                        df['possessions'] = (df['team_score'] + df['opponent_score']) / 2
                        logger.info("Estimated possessions from team scores")
                    else:
                        logger.warning("Cannot calculate possessions - missing required statistics")
                        return df
            
            # Calculate pace (possessions per 48 minutes)
            if 'minutes' in df.columns:
                df['pace'] = df['possessions'] / (df['minutes'] / 48)
            else:
                df['pace'] = df['possessions']  # Use raw possessions if minutes not available
            
            # Calculate pace-adjusted statistics
            pace_adjusted_cols = [
                'points', 'assists', 'rebounds', 'steals', 'blocks',
                'turnovers', 'field_goals_made', 'field_goals_attempted',
                'three_pointers_made', 'three_pointers_attempted',
                'free_throws_made', 'free_throws_attempted'
            ]
            
            # Only process columns that exist in the DataFrame
            existing_cols = [col for col in pace_adjusted_cols if col in df.columns]
            
            for col in existing_cols:
                # Calculate per 100 possessions
                df[f'{col}_per_100'] = df[col] / df['possessions'] * 100
                
                # Calculate per 48 minutes if minutes available
                if 'minutes' in df.columns:
                    df[f'{col}_per_48'] = df[col] / df['minutes'] * 48
                
                # Calculate pace-adjusted rate
                df[f'{col}_pace_adj'] = df[col] / df['pace'] * 100
            
            # Calculate advanced pace-adjusted metrics
            if all(col in df.columns for col in ['points', 'field_goals_attempted', 'free_throws_attempted']):
                # Pace-adjusted offensive rating
                df['ortg_pace_adj'] = df['points'] / df['pace'] * 100
                
                # Pace-adjusted true shooting attempts
                df['tsa_pace_adj'] = (df['field_goals_attempted'] + 0.44 * df['free_throws_attempted']) / df['pace'] * 100
            
            logger.info("Successfully calculated pace-adjusted statistics")
            return df
        except Exception as e:
            logger.error(f"Error calculating pace-adjusted statistics: {str(e)}")
            return df

    def process_team_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Process and combine team-level features.
        
        Args:
            df: Optional input DataFrame. If None, uses self.games_df.
            
        Returns:
            pd.DataFrame: Processed team features
        """
        try:
            # Use input DataFrame or self.games_df
            if df is None:
                df = self.games_df.copy()
            else:
                df = df.copy()
            
            if df.empty:
                self.logger.warning("No team data available for processing")
                return pd.DataFrame()
            
            # Create home team records
            home_games = df.copy()
            home_games['TEAM_ID'] = home_games['HOME_TEAM_ID']
            home_games['OPPONENT_TEAM_ID'] = home_games['AWAY_TEAM_ID']
            home_games['TEAM_ABBR'] = home_games['HOME_TEAM_ABBR']
            home_games['OPPONENT_TEAM_ABBR'] = home_games['AWAY_TEAM_ABBR']
            home_games['IS_HOME'] = 1
            
            # Create away team records
            away_games = df.copy()
            away_games['TEAM_ID'] = away_games['AWAY_TEAM_ID']
            away_games['OPPONENT_TEAM_ID'] = away_games['HOME_TEAM_ID']
            away_games['TEAM_ABBR'] = away_games['AWAY_TEAM_ABBR']
            away_games['OPPONENT_TEAM_ABBR'] = away_games['HOME_TEAM_ABBR']
            away_games['IS_HOME'] = 0
            
            # Combine home and away records
            team_features = pd.concat([home_games, away_games], ignore_index=True)
            
            # Sort by date
            if 'GAME_DATE' in team_features.columns:
                team_features = team_features.sort_values(['GAME_DATE', 'GAME_ID'])
            
            # Add advanced team stats if available
            if not self.advanced_team_df.empty:
                team_features = team_features.merge(
                    self.advanced_team_df,
                    on=['GAME_ID', 'TEAM_ID'],
                    how='left'
                )
            
            # Add tracking team stats if available
            if not self.tracking_team_df.empty:
                team_features = team_features.merge(
                    self.tracking_team_df,
                    on=['TEAM_ID', 'GAME_ID'],
                    how='left',
                    suffixes=('', '_track')
                )
            
            # Calculate efficiency metrics
            team_features = self.calculate_efficiency_metrics(team_features)
            
            # Calculate pace-adjusted statistics
            team_features = self.calculate_pace_adjusted_stats(team_features)
            
            # Calculate rolling statistics
            stat_cols = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PTS', 'PLUS_MINUS']
            available_cols = [col for col in stat_cols if col in team_features.columns]
            team_features = self.calculate_rolling_stats(
                team_features,
                ['TEAM_ID'],
                available_cols,
                [3, 5, 10]
            )
            
            # Calculate opponent-adjusted features
            team_features = self.calculate_opponent_adjusted_features(team_features)
            
            # Calculate game context features
            team_features = self.calculate_game_context_features(team_features)
            
            # Calculate streak features
            team_features = self.calculate_streak_features(team_features)
            
            # Calculate betting features if odds data is available
            if not self.odds_df.empty:
                team_features = self.calculate_betting_features(team_features, self.odds_df)
            
            self.logger.info(f"Successfully processed team features with shape {team_features.shape}")
            return team_features
            
        except Exception as e:
            self.logger.error(f"Error processing team features: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def process_player_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Process and combine player-level features.
        
        Args:
            df: Optional input DataFrame. If None, uses self.games_df.
            
        Returns:
            pd.DataFrame: Processed player features
        """
        try:
            # Use input DataFrame or self.games_df
            if df is None:
                df = self.games_df.copy()
            else:
                df = df.copy()
            
            if df.empty:
                self.logger.warning("No player data available for processing")
                return pd.DataFrame()
            
            # Create player records for both teams
            player_features = pd.DataFrame()
            
            # Process home team players
            if not self.advanced_player_df.empty:
                home_players = self.advanced_player_df[
                    self.advanced_player_df['GAME_ID'].isin(df['GAME_ID']) &
                    self.advanced_player_df['TEAM_ID'].isin(df['HOME_TEAM_ID'])
                ].copy()
                home_players['IS_HOME'] = 1
                player_features = pd.concat([player_features, home_players], ignore_index=True)
            
            # Process away team players
            if not self.advanced_player_df.empty:
                away_players = self.advanced_player_df[
                    self.advanced_player_df['GAME_ID'].isin(df['GAME_ID']) &
                    self.advanced_player_df['TEAM_ID'].isin(df['AWAY_TEAM_ID'])
                ].copy()
                away_players['IS_HOME'] = 0
                player_features = pd.concat([player_features, away_players], ignore_index=True)
            
            if player_features.empty:
                self.logger.warning("No player data found for the given games")
                return pd.DataFrame()
            
            # Add tracking player stats if available
            if not self.tracking_player_df.empty:
                player_features = player_features.merge(
                    self.tracking_player_df,
                    on=['PERSON_ID', 'GAME_ID'],
                    how='left',
                    suffixes=('', '_track')
                )
            
            # Add matchup data if available
            if not self.matchup_df.empty:
                player_features = player_features.merge(
                    self.matchup_df,
                    on=['PERSON_ID', 'GAME_ID'],
                    how='left',
                    suffixes=('', '_matchup')
                )
            
            # Calculate player efficiency metrics
            player_features = self.calculate_player_efficiency_metrics(player_features)
            
            # Calculate player impact metrics
            player_features = self.calculate_player_impact_metrics(player_features)
            
            # Calculate rolling statistics
            stat_cols = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PTS', 'PLUS_MINUS']
            available_cols = [col for col in stat_cols if col in player_features.columns]
            player_features = self.calculate_rolling_stats(
                player_features,
                ['PERSON_ID'],
                available_cols,
                [3, 5, 10]
            )
            
            # Calculate matchup-based features
            player_features = self.calculate_matchup_features(player_features)
            
            # Calculate rest and fatigue features
            player_features = self.calculate_player_rest_features(player_features)
            
            self.logger.info(f"Successfully processed player features with shape {player_features.shape}")
            return player_features
            
        except Exception as e:
            self.logger.error(f"Error processing player features: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def calculate_injury_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate injury impact score based on player importance."""
        try:
            if not hasattr(self, 'player_features_df'):
                logger.warning("Player features not available for injury impact calculation")
                return df
            
            # Get player importance metrics
            player_importance = self.player_features_df.groupby('team_id').agg({
                'minutes': 'mean',
                'points': 'mean',
                'plus_minus': 'mean',
                'player_efficiency_rating': 'mean'
            }).reset_index()
            
            # Calculate team averages
            team_avgs = player_importance.groupby('team_id').mean()
            
            # Calculate injury impact score
            injury_impact = pd.DataFrame()
            for team_id in df['team_id'].unique():
                team_players = self.player_features_df[self.player_features_df['team_id'] == team_id]
                if len(team_players) == 0:
                    continue
                
                # Calculate deviation from team averages
                importance_score = (
                    (team_players['minutes'] / team_avgs.loc[team_id, 'minutes']) +
                    (team_players['points'] / team_avgs.loc[team_id, 'points']) +
                    (team_players['plus_minus'] / team_avgs.loc[team_id, 'plus_minus']) +
                    (team_players['player_efficiency_rating'] / team_avgs.loc[team_id, 'player_efficiency_rating'])
                ) / 4
                
                # Sum importance scores for injured players
                injury_impact[team_id] = importance_score[team_players['status'] == 'INJ'].sum()
            
            # Merge with main dataframe
            df['injury_impact'] = df['team_id'].map(injury_impact)
            logger.info("Successfully calculated injury impact")
        except Exception as e:
            logger.error(f"Error calculating injury impact: {str(e)}")
        return df

    def calculate_opponent_adjusted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate opponent-adjusted features."""
        logger.info("Calculating opponent-adjusted features...")
        
        try:
            # Calculate opponent averages
            opponent_stats = df.groupby('OPPONENT_TEAM_ID').agg({
                'TEAM_SCORE': 'mean',
                'OPPONENT_SCORE': 'mean',
                'PACE': 'mean',
                'OFF_RTG': 'mean',
                'DEF_RTG': 'mean',
                'EFG_PCT': 'mean',
                'TOV_PCT': 'mean',
                'OREB_PCT': 'mean',
                'FTA_RATE': 'mean'
            }).reset_index()
            
            # Rename columns to indicate they are opponent averages
            opponent_stats.columns = ['OPPONENT_TEAM_ID'] + [f'OPP_{col}' for col in opponent_stats.columns[1:]]
            
            # Merge opponent stats back to main dataframe
            df = df.merge(opponent_stats, on='OPPONENT_TEAM_ID', how='left')
            
            # Calculate opponent-adjusted metrics
            df['ADJ_OFF_RTG'] = df['OFF_RTG'] - df['OPP_DEF_RTG']
            df['ADJ_DEF_RTG'] = df['DEF_RTG'] - df['OPP_OFF_RTG']
            df['ADJ_PACE'] = df['PACE'] - df['OPP_PACE']
            df['ADJ_EFG_PCT'] = df['EFG_PCT'] - df['OPP_EFG_PCT']
            df['ADJ_TOV_PCT'] = df['TOV_PCT'] - df['OPP_TOV_PCT']
            df['ADJ_OREB_PCT'] = df['OREB_PCT'] - df['OPP_OREB_PCT']
            df['ADJ_FTA_RATE'] = df['FTA_RATE'] - df['OPP_FTA_RATE']
            
            # Calculate opponent-adjusted scoring
            df['ADJ_SCORING'] = df['TEAM_SCORE'] - df['OPP_OPPONENT_SCORE']
            
            logger.info("Calculated opponent-adjusted features")
            return df
        except Exception as e:
            logger.error(f"Error calculating opponent-adjusted features: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return df

    def calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum features."""
        try:
            if 'win' not in df.columns:
                # Create win column from team_score and opponent_team_score
                df['win'] = (df['team_score'] > df['opponent_score']).astype(int)
            
            # Calculate 5-game win percentage
            df['team_momentum'] = df.groupby('team_id')['win'].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
            df['opponent_momentum'] = df.groupby('opponent_team_id')['win'].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
            df['momentum_differential'] = df['team_momentum'] - df['opponent_momentum']
            logger.info("Successfully calculated momentum features")
        except Exception as e:
            logger.error(f"Error calculating momentum features: {str(e)}")
        return df

    def calculate_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rest features."""
        try:
            if 'game_date' not in df.columns or 'opponent_team_id' not in df.columns:
                logger.warning("Missing required columns for rest features")
                return df
            
            # Convert game_date to datetime if not already
            df['game_date'] = pd.to_datetime(df['game_date'])
            
            # Calculate days since last game
            df['team_rest_days'] = df.groupby('team_id')['game_date'].diff().dt.days
            df['opponent_rest_days'] = df.groupby('opponent_team_id')['game_date'].diff().dt.days
            
            # Calculate rest differential
            df['rest_differential'] = df['team_rest_days'] - df['opponent_rest_days']
            logger.info("Successfully calculated rest features")
        except Exception as e:
            logger.error(f"Error calculating rest features: {str(e)}")
        return df

    def calculate_timezone_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate timezone features."""
        try:
            if 'team_abbreviation' not in df.columns or 'opponent_team_abbreviation' not in df.columns:
                logger.warning("Missing team abbreviations for timezone features")
                return df
            
            # Define timezone offsets for each team
            timezone_map = {
                'ATL': -5, 'BOS': -5, 'BKN': -5, 'CHA': -5, 'CHI': -6,
                'CLE': -5, 'DAL': -6, 'DEN': -7, 'DET': -5, 'GSW': -8,
                'HOU': -6, 'IND': -5, 'LAC': -8, 'LAL': -8, 'MEM': -6,
                'MIA': -5, 'MIL': -6, 'MIN': -6, 'NOP': -6, 'NYK': -5,
                'OKC': -6, 'ORL': -5, 'PHI': -5, 'PHX': -7, 'POR': -8,
                'SAC': -8, 'SAS': -6, 'TOR': -5, 'UTA': -7, 'WAS': -5
            }
            
            # Map timezone offsets
            df['team_timezone'] = df['team_abbreviation'].map(timezone_map)
            df['opponent_timezone'] = df['opponent_team_abbreviation'].map(timezone_map)
            
            # Calculate timezone travel lag
            df['timezone_lag'] = abs(df['team_timezone'] - df['opponent_timezone'])
            logger.info("Successfully calculated timezone features")
        except Exception as e:
            logger.error(f"Error calculating timezone features: {str(e)}")
        return df

    def calculate_over_under_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate over/under prediction features."""
        try:
            if 'team_score' not in df.columns or 'opponent_score' not in df.columns:
                logger.warning("Missing score columns for over/under features")
                return df
            
            # Calculate combined scores
            df['combined_score'] = df['team_score'] + df['opponent_score']
            
            # Calculate rolling average combined scores
            for window in [3, 5, 10]:
                df[f'combined_score_{window}g'] = df.groupby('team_id')['combined_score'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            
            # Calculate points per pace
            if 'pace' in df.columns:
                df['points_per_pace'] = df['team_score'] / df['pace']
                df['opponent_points_per_pace'] = df['opponent_score'] / df['pace']
            
            # Calculate day of week trends
            df['day_of_week'] = pd.to_datetime(df['game_date']).dt.day_name()
            day_avgs = df.groupby('day_of_week')['combined_score'].mean()
            df['day_of_week_avg'] = df['day_of_week'].map(day_avgs)
            
            logger.info("Successfully calculated over/under features")
        except Exception as e:
            logger.error(f"Error calculating over/under features: {str(e)}")
        return df

    def load_tracking_team_data(self) -> pd.DataFrame:
        """Load tracking team data from CSV."""
        try:
            tracking_team_path = self.raw_dir / "tracking" / "tracking_t_data.csv"
            if not tracking_team_path.exists():
                logger.warning("No tracking team data found")
                return pd.DataFrame()
            
            df = pd.read_csv(tracking_team_path)
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Extract game_id from game_date and team_id if not present
            if 'game_id' not in df.columns:
                logger.info("Generating game_id from game_date and team_id")
                df['game_date'] = pd.to_datetime(df['game_date'])
                df['game_id'] = df['game_date'].dt.strftime('%Y%m%d') + '_' + df['team_id'].astype(str)
            
            # Ensure proper data types
            df['game_id'] = df['game_id'].astype(str)
            df['team_id'] = pd.to_numeric(df['team_id'], errors='coerce').astype('Int64')
            
            logger.info(f"Loaded tracking team data with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading tracking team data: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return pd.DataFrame()

    def load_player_features(self) -> None:
        """Load and process player features."""
        try:
            # Load core player data
            core_path = os.path.join(self.raw_dir, "core", "nba_player_ids.csv")
            if os.path.exists(core_path):
                self.core_player_df = pd.read_csv(core_path)
                self.logger.info(f"Loaded core player data with {len(self.core_player_df)} players")
            else:
                self.logger.error("Core player data file not found")
                self.core_player_df = pd.DataFrame()
                raise ValueError("Core player data is required")
            
            # Initialize player features
            self.player_features_df = self.core_player_df.copy()
            self.logger.info("Initialized player features from core player data")
            
            # Load advanced player data if available
            adv_path = os.path.join(self.raw_dir, "advanced", "player_advanced.csv")
            if os.path.exists(adv_path):
                self.advanced_player_df = pd.read_csv(adv_path)
                self.logger.info(f"Loaded advanced player data with {len(self.advanced_player_df)} records")
                
                # Merge with core data
                if not self.player_features_df.empty:
                    self.player_features_df = self.player_features_df.merge(
                        self.advanced_player_df,
                        on=['PLAYER_ID'],
                        how='left'
                    )
            else:
                self.logger.warning("Advanced player data not found")
                self.advanced_player_df = pd.DataFrame()
            
            # Load tracking player data if available
            tracking_path = os.path.join(self.raw_dir, "tracking", "tracking_p_data.csv")
            if os.path.exists(tracking_path):
                self.tracking_player_df = pd.read_csv(tracking_path)
                self.logger.info(f"Loaded tracking player data with {len(self.tracking_player_df)} records")
                
                # Merge with existing data
                if not self.player_features_df.empty:
                    self.player_features_df = self.player_features_df.merge(
                        self.tracking_player_df,
                        on=['PLAYER_ID'],
                        how='left'
                    )
            else:
                self.logger.warning("Tracking player data not found")
                self.tracking_player_df = pd.DataFrame()
            
            self.logger.info("Successfully loaded player features")
            
        except Exception as e:
            self.logger.error(f"Error loading player features: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def analyze_feature_importance(self, df: pd.DataFrame, target_col: str, 
                                 exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Analyze feature importance using XGBoost.
        
        Args:
            df: DataFrame containing features and target
            target_col: Name of the target column
            exclude_cols: List of columns to exclude from analysis
            
        Returns:
            DataFrame with feature importance scores
        """
        try:
            # Exclude specified columns
            if exclude_cols is None:
                exclude_cols = []
            exclude_cols.append(target_col)
            
            # Get feature columns
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Split data
            X = df[feature_cols]
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost model
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Calculate cumulative importance
            importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
            
            # Add importance category
            importance_df['importance_category'] = pd.cut(
                importance_df['importance'],
                bins=[0, 0.01, 0.05, 1],
                labels=['Low', 'Medium', 'High']
            )
            
            logger.info("Feature importance analysis completed")
            return importance_df
            
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return pd.DataFrame()

    def validate_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate engineered features for data quality issues.
        
        Args:
            df: DataFrame containing engineered features
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'missing_values': [],
            'infinite_values': [],
            'outliers': [],
            'correlated_features': [],
            'constant_features': []
        }
        
        try:
            # Check for missing values
            missing_cols = df.columns[df.isna().any()].tolist()
            if missing_cols:
                validation_results['missing_values'] = missing_cols
                logger.warning(f"Found missing values in columns: {missing_cols}")
            
            # Check for infinite values
            inf_cols = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
            if inf_cols:
                validation_results['infinite_values'] = inf_cols
                logger.warning(f"Found infinite values in columns: {inf_cols}")
            
            # Check for outliers using IQR method
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[
                    (df[col] < Q1 - 1.5 * IQR) | 
                    (df[col] > Q3 + 1.5 * IQR)
                ][col]
                if len(outliers) > 0:
                    validation_results['outliers'].append(col)
                    logger.warning(f"Found outliers in column: {col}")
            
            # Check for highly correlated features
            corr_matrix = df[numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr = [column for column in upper.columns if any(upper[column] > 0.9)]
            if high_corr:
                validation_results['correlated_features'] = high_corr
                logger.warning(f"Found highly correlated features: {high_corr}")
            
            # Check for constant features
            constant_cols = df.columns[df.nunique() == 1].tolist()
            if constant_cols:
                validation_results['constant_features'] = constant_cols
                logger.warning(f"Found constant features: {constant_cols}")
            
            # Log validation summary
            if all(not issues for issues in validation_results.values()):
                logger.info("All features passed validation")
            else:
                logger.warning("Some features failed validation")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in feature validation: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return validation_results

    def calculate_advanced_offense_metrics(self, df):
        """Calculate advanced offensive metrics."""
        # Ensure numeric columns are properly typed
        numeric_cols = ['OFFENSIVE_RATING', 'MINUTES', 'PTS', 'AST', 'OREB', 'TOV', 'EFFECTIVE_FIELD_GOAL_PERCENTAGE', 'TRUE_SHOOTING_PERCENTAGE', 'ASSIST_PERCENTAGE', 'OFFENSIVE_REBOUND_PERCENTAGE', 'SHOT_DISTANCE', 'SHOT_MADE_FLAG']
        df = self.convert_numeric_columns(df, numeric_cols)
        
        # Offensive efficiency
        if all(col in df.columns for col in ['OFFENSIVE_RATING', 'MINUTES']):
            df['OFF_EFFICIENCY'] = df['OFFENSIVE_RATING'] / df['MINUTES'].replace(0, np.nan) * 48
        
        # Shot creation impact
        if all(col in df.columns for col in ['PULL_UP_FG_PCT', 'DRIVE_FG_PCT', 'CATCH_SHOOT_FG_PCT']):
            df['SHOT_CREATION_IMPACT'] = (
                df['PULL_UP_FG_PCT'].fillna(0) * 1.2 +
                df['DRIVE_FG_PCT'].fillna(0) * 1.5 +
                df['CATCH_SHOOT_FG_PCT'].fillna(0) * 1.0
            ) / 3
        
        # Shot efficiency
        if 'SHOT_MADE_FLAG' in df.columns:
            df['SHOT_EFFICIENCY'] = df['SHOT_MADE_FLAG'].fillna(0)
        
        # Shot selection
        if 'SHOT_DISTANCE' in df.columns:
            df['SHOT_SELECTION'] = np.where(
                df['SHOT_DISTANCE'] <= 5, 1.0,  # At rim
                np.where(
                    df['SHOT_DISTANCE'] <= 15, 0.8,  # Short mid-range
                    np.where(
                        df['SHOT_DISTANCE'] <= 23, 0.6,  # Long mid-range
                        0.4  # Three-pointers
                    )
                )
            )
        
        # Shot clock management
        if 'SHOT_CLOCK' in df.columns:
            df['SHOT_CLOCK_MANAGEMENT'] = np.where(
                df['SHOT_CLOCK'] <= 4, 0.8,  # Late clock
                np.where(
                    df['SHOT_CLOCK'] <= 12, 1.0,  # Mid clock
                    0.6  # Early clock
                )
            )
        
        # Overall offensive impact score
        impact_components = []
        if 'PTS' in df.columns:
            impact_components.append(df['PTS'] * 1.0)
        if 'AST' in df.columns:
            impact_components.append(df['AST'] * 1.5)
        if 'OREB' in df.columns:
            impact_components.append(df['OREB'] * 1.2)
        if 'TOV' in df.columns:
            impact_components.append(-df['TOV'] * 1.0)
        if 'SHOT_CREATION_IMPACT' in df.columns:
            impact_components.append(df['SHOT_CREATION_IMPACT'] * 1.5)
        if 'SHOT_EFFICIENCY' in df.columns:
            impact_components.append(df['SHOT_EFFICIENCY'] * 2.0)
        if 'SHOT_SELECTION' in df.columns:
            impact_components.append(df['SHOT_SELECTION'] * 1.5)
        if 'SHOT_CLOCK_MANAGEMENT' in df.columns:
            impact_components.append(df['SHOT_CLOCK_MANAGEMENT'] * 1.2)
        if 'OFFENSIVE_RATING' in df.columns:
            impact_components.append(df['OFFENSIVE_RATING'] / 100)  # Normalize rating
        
        if impact_components:
            df['ADVANCED_OFFENSIVE_IMPACT'] = sum(impact_components) / len(impact_components)
        
        return df
    
    def generate_chart_detail(self) -> None:
        """Generate a detailed chart summary of the feature engineering process."""
        try:
            chart_detail = {
                "Feature Engineering Summary": {
                    "Total Features Generated": {
                        "team_features": len(self.team_features_df.columns) if not self.team_features_df.empty else 0,
                        "player_features": len(self.player_features_df.columns) if not self.player_features_df.empty else 0,
                        "game_features": len(self.game_features_df.columns) if not self.game_features_df.empty else 0,
                        "betting_features": len(self.betting_features_df.columns) if not self.betting_features_df.empty else 0,
                        "shot_metrics": len(self.shot_metrics_df.columns) if not self.shot_metrics_df.empty else 0
                    },
                    "Feature Categories": {},
                    "Data Sources": {},
                    "Processing Steps": []
                }
            }
            
            # Track feature categories
            feature_categories = {
                "Basic Stats": ["PTS", "REB", "AST", "STL", "BLK", "TOV"],
                "Advanced Metrics": ["OFFENSIVE_RATING", "DEFENSIVE_RATING", "NET_RATING", "EFG_PCT"],
                "Pace Adjusted": ["pace", "PACE", "possessions"],
                "Rolling Averages": ["rolling", "trend", "momentum"],
                "Context Features": ["REST_DAYS", "BACK_TO_BACK", "TRAVEL_DISTANCE"],
                "Betting Features": ["SPREAD", "TOTAL", "MONEYLINE"]
            }
            
            # Count features by category
            for category, features in feature_categories.items():
                count = sum(1 for col in self.team_features_df.columns if any(feat in col.upper() for feat in features))
                if count > 0:
                    chart_detail["Feature Engineering Summary"]["Feature Categories"][category] = count
            
            # Track data sources
            data_sources = {
                "Core Stats": not self.games_df.empty,
                "Advanced Stats": not self.advanced_team_df.empty,
                "Tracking Data": not self.tracking_player_df.empty,
                "Player Data": not self.player_features_df.empty,
                "Betting Data": not self.odds_df.empty
            }
            
            for source, available in data_sources.items():
                chart_detail["Feature Engineering Summary"]["Data Sources"][source] = "Available" if available else "Not Available"
            
            # Track processing steps
            processing_steps = [
                "Data Loading",
                "Column Standardization",
                "Efficiency Metrics",
                "Pace Adjustment",
                "Rolling Statistics",
                "Context Features",
                "Betting Features"
            ]
            
            chart_detail["Feature Engineering Summary"]["Processing Steps"] = processing_steps
            
            # Save chart detail to file
            output_path = os.path.join(self.processed_dir, "feature_engineering_chart.json")
            with open(output_path, 'w') as f:
                json.dump(chart_detail, f, indent=4)
            
            self.logger.info(f"Chart detail saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating chart detail: {str(e)}")
            self.logger.error(traceback.format_exc())

    def run(self):
        """Run the feature engineering pipeline."""
        try:
            self.logger.info("Starting feature engineering pipeline")
            
            # Process team features
            self.logger.info("Processing team features")
            self.team_features_df = self.process_team_features(self.games_df)
            if not self.team_features_df.empty:
                output_path = os.path.join(self.processed_dir, "features", "team_features.csv")
                self.team_features_df.to_csv(output_path, index=False)
                self.logger.info(f"Saved team features to {output_path}")
            
            # Process player features
            self.logger.info("Processing player features")
            self.player_features_df = self.process_player_features(self.core_player_df)
            if not self.player_features_df.empty:
                output_path = os.path.join(self.processed_dir, "features", "player_features.csv")
                self.player_features_df.to_csv(output_path, index=False)
                self.logger.info(f"Saved player features to {output_path}")
            
            # Process game features
            self.logger.info("Processing game features")
            self.game_features_df = self.process_game_features()
            if not self.game_features_df.empty:
                output_path = os.path.join(self.processed_dir, "features", "game_features.csv")
                self.game_features_df.to_csv(output_path, index=False)
                self.logger.info(f"Saved game features to {output_path}")
            
            # Process betting features
            self.logger.info("Processing betting features")
            self.betting_features_df = self.process_betting_features(self.odds_df)
            if not self.betting_features_df.empty:
                output_path = os.path.join(self.processed_dir, "features", "betting_features.csv")
                self.betting_features_df.to_csv(output_path, index=False)
                self.logger.info(f"Saved betting features to {output_path}")
            
            # Process shot metrics if tracking data is available
            if not self.tracking_player_df.empty:
                self.logger.info("Processing shot metrics")
                self.shot_metrics_df = self.process_shot_metrics()
                if not self.shot_metrics_df.empty:
                    output_path = os.path.join(self.processed_dir, "features", "shot_metrics.csv")
                    self.shot_metrics_df.to_csv(output_path, index=False)
                    self.logger.info(f"Saved shot metrics to {output_path}")
            
            # Generate feature engineering summary
            self.logger.info("Generating feature engineering summary")
            self.generate_chart_detail()
            
            self.logger.info("Feature engineering pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def load_team_features(self) -> pd.DataFrame:
        """Load and process team features from raw data files.
        
        Returns:
            pd.DataFrame: Processed team features
        """
        try:
            # Load core team data
            core_path = os.path.join(self.raw_dir, 'core', 'team_data.csv')
            self.logger.info(f"Loading core team data from {core_path}")
            team_data = pd.read_csv(core_path)
            
            # Load advanced team data
            adv_path = os.path.join(self.raw_dir, 'advanced', 'team_advanced.csv')
            self.logger.info(f"Loading advanced team data from {adv_path}")
            team_advanced = pd.read_csv(adv_path)
            
            # Check required columns
            required_cols = ['GAME_ID', 'TEAM_ID', 'GAME_DATE']
            for df, name in [(team_data, 'core team data'), (team_advanced, 'advanced team data')]:
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing required columns in {name}: {missing}")
            
            # Convert GAME_DATE to datetime
            try:
                team_data['GAME_DATE'] = pd.to_datetime(team_data['GAME_DATE'], format='%Y%m%d')
                team_advanced['GAME_DATE'] = pd.to_datetime(team_advanced['GAME_DATE'], format='%Y%m%d')
            except Exception as e:
                self.logger.error(f"Error converting dates: {str(e)}")
                self.logger.error("Attempting flexible date parsing...")
                team_data['GAME_DATE'] = pd.to_datetime(team_data['GAME_DATE'])
                team_advanced['GAME_DATE'] = pd.to_datetime(team_advanced['GAME_DATE'])
            
            # Merge core and advanced data
            df = pd.merge(
                team_data,
                team_advanced,
                on=['GAME_ID', 'TEAM_ID', 'GAME_DATE'],
                how='inner',
                validate='1:1'
            )
            
            # Add home/away and opponent team IDs
            df['is_home'] = df.groupby('GAME_ID')['TEAM_ID'].transform('first') == df['TEAM_ID']
            df['opponent_team_id'] = df.groupby('GAME_ID')['TEAM_ID'].transform(lambda x: x.iloc[1] if x.iloc[0] == x.name else x.iloc[0])
            
            # Calculate efficiency metrics
            df = self.calculate_efficiency_metrics(df)
            
            # Calculate pace-adjusted stats if possible
            if 'POSS' in df.columns:
                pace_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
                for col in pace_cols:
                    if col in df.columns:
                        df[f'{col}_PER_100'] = df[col] / df['POSS'] * 100
            
            # Calculate net rating
            if all(col in df.columns for col in ['OFFENSIVE_RATING', 'DEFENSIVE_RATING']):
                df['NET_RATING'] = df['OFFENSIVE_RATING'] - df['DEFENSIVE_RATING']
            
            # Sort by date and team
            df = df.sort_values(['GAME_DATE', 'TEAM_ID'])
            
            # Calculate rolling stats
            rolling_windows = [3, 5, 10]
            metrics = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'NET_RATING']
            metrics = []
            
            for window in rolling_windows:
                for metric in metrics:
                    df[f'{metric}_ROLLING_{window}'] = df.groupby('TEAM_ID')[metric].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
            
            # Drop rows with missing dates
            df = df.dropna(subset=['GAME_DATE'])
            
            self.logger.info(f"Successfully loaded team features with shape {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in load_team_features: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def process_shot_metrics(self) -> pd.DataFrame:
        """Process shot chart data to generate detailed shot metrics features.
        
        Returns:
            pd.DataFrame: DataFrame containing shot metrics features
        """
        try:
            # Load tracking player data
            tracking_path = os.path.join(self.raw_dir, "tracking", "tracking_p_data.csv")
            if not os.path.exists(tracking_path):
                self.logger.error("Tracking player data file not found")
                return pd.DataFrame()
            
            shot_data = pd.read_csv(tracking_path)
            
            # Convert coordinates to numeric
            shot_data['LOC_X'] = pd.to_numeric(shot_data['LOC_X'], errors='coerce')
            shot_data['LOC_Y'] = pd.to_numeric(shot_data['LOC_Y'], errors='coerce')
            
            # Calculate shot metrics by player and game
            shot_metrics = shot_data.groupby(['PLAYER_ID', 'GAME_ID']).agg({
                'SHOT_MADE_FLAG': ['sum', 'count', 'mean'],  # FG%, total shots, makes
                'SHOT_DISTANCE': ['mean', 'std', 'min', 'max'],  # Distance metrics
                'LOC_X': ['mean', 'std'],  # Horizontal shot distribution
                'LOC_Y': ['mean', 'std'],  # Vertical shot distribution
                'MIN': 'sum',  # Minutes played
                'FGM': 'sum',  # Field goals made
                'FGA': 'sum',  # Field goals attempted
                'FG3M': 'sum',  # Three pointers made
                'FG3A': 'sum',  # Three pointers attempted
                'FTM': 'sum',  # Free throws made
                'FTA': 'sum',  # Free throws attempted
                'PTS': 'sum',  # Total points
                'CLOSEST_DEFENDER_DISTANCE': ['mean', 'min', 'max'],  # Defender pressure
                'SHOT_CLOCK': ['mean', 'std'],  # Shot clock metrics
                'TOUCH_TIME': ['mean', 'std'],  # Touch time metrics
                'DRIBBLES': ['mean', 'std'],  # Dribble metrics
                'PERIOD': 'count'  # For quarter-specific metrics
            }).reset_index()
            
            # Flatten column names
            shot_metrics.columns = ['_'.join(col).strip('_') for col in shot_metrics.columns.values]
            
            # Calculate zone-specific metrics
            zone_metrics = shot_data.groupby(['PLAYER_ID', 'GAME_ID', 'SHOT_ZONE_BASIC']).agg({
                'SHOT_MADE_FLAG': ['sum', 'count', 'mean']
            }).unstack().reset_index()
            
            # Flatten zone metrics column names
            zone_metrics.columns = ['_'.join(col).strip('_') for col in zone_metrics.columns.values]
            
            # Merge zone metrics with overall metrics
            shot_metrics = shot_metrics.merge(zone_metrics, on=['PLAYER_ID', 'GAME_ID'], how='left')
            
            # Calculate advanced shot metrics
            shot_metrics['EFFICIENT_SHOT_PCT'] = shot_metrics['SHOT_MADE_FLAG_sum'] / shot_metrics['SHOT_MADE_FLAG_count']
            shot_metrics['FG_PCT'] = shot_metrics['FGM'] / shot_metrics['FGA']
            shot_metrics['FG3_PCT'] = shot_metrics['FG3M'] / shot_metrics['FG3A']
            shot_metrics['FT_PCT'] = shot_metrics['FTM'] / shot_metrics['FTA']
            
            # Calculate shot distribution metrics
            total_shots = shot_metrics['SHOT_MADE_FLAG_count']
            for zone in ['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range', 'Left Corner 3', 'Right Corner 3', 'Above the Break 3']:
                if f'SHOT_MADE_FLAG_count_{zone}' in shot_metrics.columns:
                    shot_metrics[f'{zone.replace(" ", "_")}_SHOT_PCT'] = (
                        shot_metrics[f'SHOT_MADE_FLAG_count_{zone}'] / total_shots
                    )
            
            # Calculate shot quality metrics
            shot_metrics['SHOT_QUALITY'] = (
                shot_metrics['SHOT_MADE_FLAG_mean'] * 
                (1 - shot_metrics['SHOT_DISTANCE_mean'] / 30)  # Normalize by max distance
            )
            
            # Calculate shot difficulty based on defender pressure
            shot_metrics['SHOT_DIFFICULTY'] = (
                shot_metrics['CLOSEST_DEFENDER_DISTANCE_mean'] / 
                shot_metrics['SHOT_DISTANCE_mean']
            )
            
            # Calculate shot clock pressure metrics
            shot_metrics['SHOT_CLOCK_PRESSURE'] = (
                shot_metrics['SHOT_CLOCK_mean'] / 24  # Normalize by shot clock length
            )
            
            # Calculate shot creation metrics
            shot_metrics['SHOT_CREATION'] = (
                shot_metrics['TOUCH_TIME_mean'] * 
                shot_metrics['DRIBBLES_mean']
            )
            
            # Calculate catch-and-shoot vs pull-up metrics
            shot_metrics['CATCH_SHOOT_PCT'] = (
                shot_data[shot_data['TOUCH_TIME'] < 2].groupby(['PLAYER_ID', 'GAME_ID'])['SHOT_MADE_FLAG'].mean()
            ).reset_index()['SHOT_MADE_FLAG']
            
            shot_metrics['PULL_UP_PCT'] = (
                shot_data[shot_data['TOUCH_TIME'] >= 2].groupby(['PLAYER_ID', 'GAME_ID'])['SHOT_MADE_FLAG'].mean()
            ).reset_index()['SHOT_MADE_FLAG']
            
            # Calculate quarter-specific metrics
            for quarter in range(1, 5):
                quarter_data = shot_data[shot_data['PERIOD'] == quarter]
                quarter_metrics = quarter_data.groupby(['PLAYER_ID', 'GAME_ID']).agg({
                    'SHOT_MADE_FLAG': ['sum', 'count', 'mean']
                }).reset_index()
                quarter_metrics.columns = ['PLAYER_ID', 'GAME_ID', f'Q{quarter}_FGM', f'Q{quarter}_FGA', f'Q{quarter}_FG_PCT']
                shot_metrics = shot_metrics.merge(quarter_metrics, on=['PLAYER_ID', 'GAME_ID'], how='left')
            
            # Calculate clutch shooting metrics (last 5 minutes of close games)
            clutch_data = shot_data[
                (shot_data['PERIOD'] >= 4) & 
                (shot_data['GAME_CLOCK'] <= '05:00') & 
                (abs(shot_data['SCORE_MARGIN']) <= 5)
            ]
            clutch_metrics = clutch_data.groupby(['PLAYER_ID', 'GAME_ID']).agg({
                'SHOT_MADE_FLAG': ['sum', 'count', 'mean']
            }).reset_index()
            clutch_metrics.columns = ['PLAYER_ID', 'GAME_ID', 'CLUTCH_FGM', 'CLUTCH_FGA', 'CLUTCH_FG_PCT']
            shot_metrics = shot_metrics.merge(clutch_metrics, on=['PLAYER_ID', 'GAME_ID'], how='left')
            
            # Calculate shot selection efficiency
            shot_metrics['SHOT_SELECTION_EFFICIENCY'] = (
                shot_metrics['SHOT_MADE_FLAG_mean'] * 
                (1 - shot_metrics['SHOT_DISTANCE_mean'] / 30) *  # Distance factor
                (1 - shot_metrics['CLOSEST_DEFENDER_DISTANCE_mean'] / 10)  # Defender factor
            )
            
            # Calculate points per shot
            shot_metrics['POINTS_PER_SHOT'] = shot_metrics['PTS'] / shot_metrics['SHOT_MADE_FLAG_count']
            
            # Calculate true shooting percentage
            shot_metrics['TRUE_SHOOTING_PCT'] = (
                shot_metrics['PTS'] / 
                (2 * (shot_metrics['FGA'] + 0.44 * shot_metrics['FTA']))
            )
            
            # Calculate effective field goal percentage
            shot_metrics['EFG_PCT'] = (
                (shot_metrics['FGM'] + 0.5 * shot_metrics['FG3M']) / 
                shot_metrics['FGA']
            )
            
            # Calculate shot selection metrics
            shot_metrics['THREE_POINT_RATE'] = shot_metrics['FG3A'] / shot_metrics['FGA']
            shot_metrics['FREE_THROW_RATE'] = shot_metrics['FTA'] / shot_metrics['FGA']
            
            # Calculate shot creation metrics
            shot_metrics['SHOT_CREATION'] = (
                shot_metrics['SHOT_MADE_FLAG_count'] / 
                shot_metrics['MIN'] * 48  # Normalize to per 48 minutes
            )
            
            # Save to CSV
            output_path = os.path.join(self.processed_dir, "features", "shot_metrics.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shot_metrics.to_csv(output_path, index=False)
            
            self.logger.info(f"Shot metrics saved to {output_path}")
            return shot_metrics
            
        except Exception as e:
            self.logger.error(f"Error processing shot metrics: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def process_game_features(self) -> pd.DataFrame:
        """Process game-level features including context, momentum, and matchup data."""
        try:
            if self.games_df.empty:
                self.logger.warning("No games data available for game features")
                return pd.DataFrame()
            
            # Create a copy to avoid modifying the original
            game_features = self.games_df.copy()
            
            # Add game context features
            game_features = self.calculate_game_context_features(game_features)
            
            # Add momentum features
            game_features = self.calculate_momentum_features(game_features)
            
            # Add rest features
            game_features = self.calculate_rest_features(game_features)
            
            # Add timezone features
            game_features = self.calculate_timezone_features(game_features)
            
            # Add matchup features
            game_features = self.calculate_matchup_features(game_features)
            
            # Add over/under features
            game_features = self.calculate_over_under_features(game_features)
            
            # Add streak features
            game_features = self.calculate_streak_features(game_features)
            
            # Add head-to-head features
            game_features = self.calculate_h2h_features(game_features)
            
            # Add injury impact if available
            if not self.player_features_df.empty:
                game_features = self.calculate_injury_impact(game_features)
            
            return game_features
            
        except Exception as e:
            self.logger.error(f"Error processing game features: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def visualize_feature_importance(self, target_col: str, output_dir: str = None) -> None:
        """Generate and save feature importance visualizations.
        
        Args:
            target_col: Target column for feature importance analysis
            output_dir: Directory to save visualizations (defaults to features_dir)
        """
        try:
            if output_dir is None:
                output_dir = self.features_dir
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Analyze feature importance for each feature set
            feature_sets = {
                'team': self.team_features_df,
                'player': self.player_features_df,
                'game': self.game_features_df,
                'betting': self.betting_features_df,
                'shot': self.shot_metrics_df
            }
            
            for feature_type, df in feature_sets.items():
                if df.empty or target_col not in df.columns:
                    continue
                
                # Get feature importance
                importance_df = self.analyze_feature_importance(df, target_col)
                
                if importance_df.empty:
                    continue
                
                # Create visualization
                plt.figure(figsize=(12, 8))
                sns.barplot(
                    data=importance_df.head(20),
                    x='importance',
                    y='feature',
                    hue='importance_category'
                )
                plt.title(f'Top 20 {feature_type.title()} Features by Importance')
                plt.xlabel('Feature Importance')
                plt.ylabel('Feature Name')
                plt.tight_layout()
                
                # Save visualization
                output_path = os.path.join(output_dir, f'{feature_type}_feature_importance.png')
                plt.savefig(output_path)
                plt.close()
                
                self.logger.info(f"Saved {feature_type} feature importance visualization to {output_path}")
                
                # Save importance data
                importance_path = os.path.join(output_dir, f'{feature_type}_feature_importance.csv')
                importance_df.to_csv(importance_path, index=False)
                self.logger.info(f"Saved {feature_type} feature importance data to {importance_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating feature importance visualizations: {str(e)}")
            self.logger.error(traceback.format_exc())

    def validate_features(self) -> Dict[str, List[str]]:
        """Validate all feature sets for data quality issues.
        
        Returns:
            Dictionary containing validation results for each feature set
        """
        validation_results = {}
        
        try:
            feature_sets = {
                'team': self.team_features_df,
                'player': self.player_features_df,
                'game': self.game_features_df,
                'betting': self.betting_features_df,
                'shot': self.shot_metrics_df
            }
            
            for feature_type, df in feature_sets.items():
                if df.empty:
                    continue
                
                validation_results[feature_type] = self._validate_feature_set(df)
            
            # Save validation results
            validation_path = os.path.join(self.features_dir, "feature_validation.json")
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=4)
            
            self.logger.info(f"Saved feature validation results to {validation_path}")
            
        except Exception as e:
            self.logger.error(f"Error validating features: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        return validation_results

    def _validate_feature_set(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate a single feature set for data quality issues.
        
        Args:
            df: DataFrame containing features to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'missing_values': [],
            'infinite_values': [],
            'outliers': [],
            'correlated_features': [],
            'constant_features': []
        }
        
        try:
            # Check for missing values
            missing_cols = df.columns[df.isna().any()].tolist()
            if missing_cols:
                validation_results['missing_values'] = missing_cols
            
            # Check for infinite values
            inf_cols = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
            if inf_cols:
                validation_results['infinite_values'] = inf_cols
            
            # Check for outliers using IQR method
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[
                    (df[col] < Q1 - 1.5 * IQR) | 
                    (df[col] > Q3 + 1.5 * IQR)
                ][col]
                if len(outliers) > 0:
                    validation_results['outliers'].append(col)
            
            # Check for highly correlated features
            corr_matrix = df[numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr = [column for column in upper.columns if any(upper[column] > 0.9)]
            if high_corr:
                validation_results['correlated_features'] = high_corr
            
            # Check for constant features
            constant_cols = df.columns[df.nunique() == 1].tolist()
            if constant_cols:
                validation_results['constant_features'] = constant_cols
            
        except Exception as e:
            self.logger.error(f"Error validating feature set: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        return validation_results

    def calculate_player_efficiency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate player efficiency metrics.
        
        Args:
            df: DataFrame containing player statistics
            
        Returns:
            pd.DataFrame: DataFrame with added efficiency metrics
        """
        try:
            # Calculate possessions
            df['POSS'] = 0.96 * (
                df['FGA'] + 
                0.44 * df['FTA'] - 
                df['OREB'] + 
                df['TOV']
            )
            
            # Calculate per-possession stats
            pace_stats = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV']
            for stat in pace_stats:
                if stat in df.columns:
                    df[f'{stat}_PER_POSS'] = df[stat] / df['POSS'] * 100
            
            # Calculate shooting efficiency
            if all(col in df.columns for col in ['FGM', 'FGA', 'FG3M']):
                df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
            
            if all(col in df.columns for col in ['PTS', 'FGA', 'FTA']):
                df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating player efficiency metrics: {str(e)}")
            return df

    def calculate_player_impact_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate player impact metrics.
        
        Args:
            df: DataFrame containing player statistics
            
        Returns:
            pd.DataFrame: DataFrame with added impact metrics
        """
        try:
            # Calculate usage rate
            if all(col in df.columns for col in ['FGA', 'FTA', 'TOV', 'POSS']):
                df['USG_PCT'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['POSS'] * 100
            
            # Calculate assist percentage
            if all(col in df.columns for col in ['AST', 'FGM']):
                df['AST_PCT'] = df['AST'] / df['FGM'] * 100
            
            # Calculate assist-to-turnover ratio
            if all(col in df.columns for col in ['AST', 'TOV']):
                df['AST_TO_TOV'] = df['AST'] / df['TOV'].replace(0, 1)
            
            # Calculate rebound percentages
            if all(col in df.columns for col in ['OREB', 'DREB', 'MIN']):
                df['OREB_PCT'] = df['OREB'] / df['MIN'] * 48
                df['DREB_PCT'] = df['DREB'] / df['MIN'] * 48
                df['REB_PCT'] = (df['OREB'] + df['DREB']) / df['MIN'] * 48
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating player impact metrics: {str(e)}")
            return df

    def calculate_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate matchup-based features.
        
        Args:
            df: DataFrame containing player statistics and matchup data
            
        Returns:
            pd.DataFrame: DataFrame with added matchup features
        """
        try:
            # Calculate matchup efficiency
            if all(col in df.columns for col in ['MATCHUP_PTS', 'MATCHUP_MIN']):
                df['MATCHUP_PTS_PER_MIN'] = df['MATCHUP_PTS'] / df['MATCHUP_MIN'].replace(0, 1)
            
            # Calculate matchup shooting percentages
            if all(col in df.columns for col in ['MATCHUP_FGM', 'MATCHUP_FGA']):
                df['MATCHUP_FG_PCT'] = df['MATCHUP_FGM'] / df['MATCHUP_FGA'].replace(0, 1)
            
            if all(col in df.columns for col in ['MATCHUP_FG3M', 'MATCHUP_FG3A']):
                df['MATCHUP_FG3_PCT'] = df['MATCHUP_FG3M'] / df['MATCHUP_FG3A'].replace(0, 1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating matchup features: {str(e)}")
            return df

    def calculate_player_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rest and fatigue features.
        
        Args:
            df: DataFrame containing player statistics
            
        Returns:
            pd.DataFrame: DataFrame with added rest features
        """
        try:
            # Sort by player and date
            df = df.sort_values(['PERSON_ID', 'GAME_DATE'])
            
            # Calculate days since last game
            df['DAYS_SINCE_LAST_GAME'] = df.groupby('PERSON_ID')['GAME_DATE'].diff().dt.days
            
            # Calculate minutes played in last N games
            for n in [1, 2, 3]:
                df[f'MIN_LAST_{n}_GAMES'] = df.groupby('PERSON_ID')['MIN'].transform(
                    lambda x: x.rolling(n, min_periods=1).sum()
                )
            
            # Calculate back-to-back indicator
            df['IS_BACK_TO_BACK'] = (df['DAYS_SINCE_LAST_GAME'] == 1).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating player rest features: {str(e)}")
            return df

    def process_betting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process betting-related features.
        
        Args:
            df: Input DataFrame containing odds data
            
        Returns:
            pd.DataFrame: Processed betting features
        """
        try:
            if df.empty:
                self.logger.warning("No betting data available for processing")
                return pd.DataFrame()
            
            # Create home/away indicator
            df['IS_HOME'] = (df['HOME_TEAM_ID'] == df['TEAM_ID']).astype(int)
            
            # Ensure game_id is string type
            df['GAME_ID'] = df['GAME_ID'].astype(str)
            
            # Calculate implied probabilities from moneyline odds
            def moneyline_to_prob(moneyline):
                if pd.isna(moneyline):
                    return np.nan
                if moneyline > 0:
                    return 100 / (moneyline + 100)
                else:
                    return (-moneyline) / (-moneyline + 100)
            
            # Calculate implied probabilities
            if 'HOME_MONEYLINE' in df.columns and 'AWAY_MONEYLINE' in df.columns:
                df['IMPLIED_WIN_PROB'] = np.where(
                    df['IS_HOME'] == 1,
                    df['HOME_MONEYLINE'].apply(moneyline_to_prob),
                    df['AWAY_MONEYLINE'].apply(moneyline_to_prob)
                )
            
            # Calculate spread for each team
            if 'HOME_SPREAD' in df.columns and 'AWAY_SPREAD' in df.columns:
                df['SPREAD'] = np.where(
                    df['IS_HOME'] == 1,
                    df['HOME_SPREAD'],
                    df['AWAY_SPREAD']
                )
            
            # Calculate if team covered the spread
            if all(col in df.columns for col in ['TEAM_SCORE', 'OPPONENT_SCORE', 'SPREAD']):
                df['COVERED_SPREAD'] = (df['TEAM_SCORE'] - df['OPPONENT_SCORE']) > df['SPREAD']
            
            # Calculate over/under result
            if all(col in df.columns for col in ['TEAM_SCORE', 'OPPONENT_SCORE', 'TOTAL']):
                df['TOTAL_POINTS'] = df['TEAM_SCORE'] + df['OPPONENT_SCORE']
                df['OVER'] = df['TOTAL_POINTS'] > df['TOTAL']
            
            # Calculate rolling betting performance
            if 'COVERED_SPREAD' in df.columns:
                for window in [3, 5, 10]:
                    # Full game rolling metrics
                    df[f'COVER_RATE_{window}G'] = df.groupby('TEAM_ID')['COVERED_SPREAD'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    if 'OVER' in df.columns:
                        df[f'OVER_RATE_{window}G'] = df.groupby('TEAM_ID')['OVER'].transform(
                            lambda x: x.rolling(window, min_periods=1).mean()
                        )
            
            # Calculate market efficiency metrics
            if all(col in df.columns for col in ['IMPLIED_WIN_PROB', 'WIN']):
                df['PREDICTION_ERROR'] = df['IMPLIED_WIN_PROB'] - df['WIN']
            if all(col in df.columns for col in ['SPREAD', 'TEAM_SCORE', 'OPPONENT_SCORE']):
                df['SPREAD_ERROR'] = df['SPREAD'] - (df['TEAM_SCORE'] - df['OPPONENT_SCORE'])
            if all(col in df.columns for col in ['TOTAL', 'TOTAL_POINTS']):
                df['TOTAL_ERROR'] = df['TOTAL'] - df['TOTAL_POINTS']
            
            self.logger.info(f"Successfully processed betting features with shape {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing betting features: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

if __name__ == "__main__":
    try:
        engineer = FeatureEngineer()
        engineer.run() 
    except Exception as e:
        logger.error(f"Error running feature engineering script: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise 