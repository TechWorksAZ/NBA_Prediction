"""
NBA Data Processing Pipeline

This script processes raw NBA data into clean, feature-rich datasets for model training.
It handles data from multiple sources and creates processed datasets with engineered features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json
import glob
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        """Initialize the data processor with paths and configurations."""
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / 'data'
        self.RAW_DIR = self.DATA_DIR / 'raw'
        self.PROCESSED_DIR = self.DATA_DIR / 'processed'
        self.team_mapping = self._load_team_mapping()
        self.validation_rules = self._load_validation_rules()
        
    def _load_team_mapping(self) -> Dict:
        """Load team mapping from JSON file."""
        with open(self.RAW_DIR / 'team_mapping.json', 'r') as f:
            return json.load(f)
            
    def _load_validation_rules(self) -> Dict:
        """Load data validation rules from JSON file."""
        with open(self.RAW_DIR / 'validation_rules.json', 'r') as f:
            return json.load(f)
            
    def _read_csv(self, filename: str) -> pd.DataFrame:
        """Helper method to read CSV files with error handling."""
        try:
            return pd.read_csv(self.RAW_DIR / 'core' / filename)
        except Exception as e:
            logger.error(f"Error reading {filename}: {str(e)}")
            raise
            
    def _read_advanced_csv(self, filename: str) -> pd.DataFrame:
        """Helper method to read advanced stats CSV files."""
        try:
            return pd.read_csv(self.RAW_DIR / 'advanced' / filename)
        except Exception as e:
            logger.error(f"Error reading advanced stats {filename}: {str(e)}")
            raise
            
    def _read_api_data(self, api_type: str) -> pd.DataFrame:
        """Helper method to read API data files."""
        try:
            files = glob.glob(str(self.RAW_DIR / 'advanced' / api_type / '*.csv'))
            dfs = []
            for file in files:
                df = pd.read_csv(file)
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        except Exception as e:
            logger.error(f"Error reading API data {api_type}: {str(e)}")
            raise
            
    def _read_betting_csv(self, filename: str) -> pd.DataFrame:
        """Helper method to read betting data CSV files."""
        try:
            return pd.read_csv(self.RAW_DIR / 'betting' / filename)
        except Exception as e:
            logger.error(f"Error reading betting data {filename}: {str(e)}")
            raise
            
    def _read_sbr_daily(self) -> pd.DataFrame:
        """Helper method to read SBR daily odds files."""
        try:
            files = glob.glob(str(self.RAW_DIR / 'betting' / 'sbr_daily' / '*.csv'))
            dfs = []
            for file in files:
                df = pd.read_csv(file)
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        except Exception as e:
            logger.error(f"Error reading SBR daily odds: {str(e)}")
            raise
            
    def _read_matchup_data(self, matchup_type: str) -> pd.DataFrame:
        """Helper method to read matchup data files."""
        try:
            files = glob.glob(str(self.RAW_DIR / 'matchups' / matchup_type / '*.csv'))
            dfs = []
            for file in files:
                df = pd.read_csv(file)
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        except Exception as e:
            logger.error(f"Error reading matchup data {matchup_type}: {str(e)}")
            raise
            
    def _read_defense_data(self, defense_type: str) -> pd.DataFrame:
        """Helper method to read defense data files."""
        try:
            files = glob.glob(str(self.RAW_DIR / 'defense' / defense_type / '*.csv'))
            dfs = []
            for file in files:
                df = pd.read_csv(file)
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        except Exception as e:
            logger.error(f"Error reading defense data {defense_type}: {str(e)}")
            raise
            
    def _read_tracking_data(self, tracking_type: str) -> pd.DataFrame:
        """Helper method to read tracking data files."""
        try:
            if tracking_type == 'shotchartdetail':
                files = glob.glob(str(self.RAW_DIR / 'tracking' / tracking_type / '*.csv'))
                dfs = []
                for file in files:
                    df = pd.read_csv(file)
                    dfs.append(df)
                return pd.concat(dfs, ignore_index=True)
            else:
                return pd.read_csv(self.RAW_DIR / 'tracking' / f'{tracking_type}_2024-25.csv')
        except Exception as e:
            logger.error(f"Error reading tracking data {tracking_type}: {str(e)}")
            raise

    def _calculate_rolling_stats(
        self,
        df: pd.DataFrame,
        columns_to_roll: List[str],
        window_sizes: List[int]
    ) -> pd.DataFrame:
        """
        Calculate rolling statistics for specified columns and window sizes.
        
        Args:
            df: DataFrame with team statistics
            columns_to_roll: List of columns to calculate rolling stats for
            window_sizes: List of window sizes for rolling calculations
            
        Returns:
            DataFrame with rolling statistics
        """
        logger.debug(f"Calculating rolling stats for columns: {columns_to_roll}")
        
        # Set index for rolling calculations
        df = df.set_index(['team_id', 'game_date'])
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Calculate rolling statistics for each window size
        for window in window_sizes:
            # Calculate rolling mean and std for specified columns
            rolling_stats = df[columns_to_roll].groupby('team_id').rolling(
                window=window,
                min_periods=1
            ).agg(['mean', 'std'])
            
            # Flatten column names
            rolling_stats.columns = [
                f"{col[0]}_{window}g_{col[1]}"
                for col in rolling_stats.columns
            ]
            
            # Add to result
            result = pd.concat([result, rolling_stats], axis=1)
        
        # Reset index to get team_id and game_date back as columns
        result = result.reset_index()
        
        logger.debug(f"Completed rolling stats calculation")
        return result

    def _validate_core_stats(self, df: pd.DataFrame) -> None:
        """Validate processed core statistics.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        required_columns = [
            'game_id', 'game_date', 'team_id', 'team_abbreviation',
            'team_score', 'opponent_team_score', 'home_team', 'away_team'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for null values in key columns
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Null values found in key columns:\n{null_counts[null_counts > 0]}")
        
        # Validate data types
        if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
            raise ValueError("game_date must be datetime type")
            
        if not pd.api.types.is_numeric_dtype(df['team_score']):
            raise ValueError("team_score must be numeric type")
            
        if not pd.api.types.is_numeric_dtype(df['opponent_team_score']):
            raise ValueError("opponent_team_score must be numeric type")
        
        # Validate team IDs and abbreviations
        if not df['team_id'].isin(self.team_mapping['team_id']).all():
            raise ValueError("Invalid team_id values found")
            
        if not df['team_abbreviation'].isin(self.team_mapping['team_abbreviation']).all():
            raise ValueError("Invalid team_abbreviation values found")
        
        # Validate game scores
        if (df['team_score'] < 0).any():
            raise ValueError("Negative team scores found")
            
        if (df['opponent_team_score'] < 0).any():
            raise ValueError("Negative opponent scores found")
        
        logger.info("Core statistics validation passed")

    def process_core_stats(self) -> pd.DataFrame:
        """Process core team statistics."""
        logger.info("Processing core statistics...")
        
        # Read data
        logger.info("Reading team stats...")
        team_stats = self._read_csv('core/nba_team_stats.csv')
        team_box_scores = self._read_csv('core/nba_team_box_scores.csv')
        schedule = self._read_csv('core/nba_schedule.csv')
        
        # Check required columns
        required_columns = {
            'team_stats': ['game_id', 'game_date', 'team_id', 'team_abbreviation'],
            'box_scores': ['game_id', 'game_date', 'team_id', 'team_abbreviation', 'team_score', 'opponent_team_score'],
            'schedule': ['game_id', 'game_date', 'home_team', 'away_team']
        }
        
        for df_name, required in required_columns.items():
            df = locals()[df_name]
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns in {df_name}: {missing}")
        
        # Process dates
        for df in [team_stats, team_box_scores, schedule]:
            df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Calculate rolling statistics
        logger.info("Calculating rolling statistics...")
        team_box_scores = self._calculate_rolling_stats(
            team_box_scores,
            columns_to_roll=['team_score', 'opponent_team_score'],
            window_sizes=[3, 5, 10]
        )
        
        # Merge data
        logger.info("Merging data...")
        # First merge box scores with team stats
        merged = pd.merge(
            team_box_scores,
            team_stats,
            on=['game_id', 'game_date', 'team_id', 'team_abbreviation'],
            how='left'
        )
        
        # Then merge with schedule
        merged = pd.merge(
            merged,
            schedule,
            on=['game_id', 'game_date'],
            how='left'
        )
        
        # Validate processed data
        logger.info("Validating processed data...")
        self._validate_core_stats(merged)
        
        logger.info("Core statistics processing completed successfully")
        return merged

    def process_advanced_stats(self) -> pd.DataFrame:
        """Process all advanced statistics data.
        
        Returns:
            pd.DataFrame: Processed advanced statistics
        """
        logger.info("Processing advanced statistics...")
        
        # Read and merge different types of advanced stats
        advanced_types = [
            'boxscoreadvancedv3',
            'boxscorefourfactorv3',
            'boxscoremiscv3',
            'boxscoreplayertrackv3',
            'boxscorescoringv3',
            'boxscoreusagev3'
        ]
        
        dfs = []
        for adv_type in advanced_types:
            df = self._read_advanced_data(adv_type)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            logger.error("No advanced statistics data found")
            return pd.DataFrame()
        
        # Merge all advanced stats
        advanced_stats = pd.concat(dfs, ignore_index=True)
        
        # Process and clean the data
        advanced_stats['game_date'] = pd.to_datetime(advanced_stats['game_date'])
        advanced_stats = advanced_stats.sort_values(['team_id', 'game_date'])
        
        # Calculate additional metrics
        advanced_stats['net_rating'] = advanced_stats['off_rating'] - advanced_stats['def_rating']
        advanced_stats['true_shooting_pct'] = advanced_stats['ts_pct']
        
        # Calculate rolling statistics
        stats_to_roll = [
            'off_rating', 'def_rating', 'net_rating',
            'ast_pct', 'oreb_pct', 'dreb_pct',
            'tm_tov_pct', 'efg_pct', 'ts_pct',
            'pace'
        ]
        
        rolling_stats = self._calculate_rolling_stats(
            advanced_stats,
            stats_to_roll,
            [3, 5, 10]
        )
        
        # Merge rolling stats
        advanced_stats = pd.merge(
            advanced_stats,
            rolling_stats,
            on=['team_id', 'game_date'],
            how='left'
        )
        
        # Validate the processed data
        self._validate_advanced_stats(advanced_stats)
        
        logger.info("Advanced statistics processing complete")
        return advanced_stats
        
    def process_betting_data(self) -> pd.DataFrame:
        """Process betting data from SBR odds files.
        
        Returns:
            pd.DataFrame: Processed betting data with calculated metrics
        """
        logger.info("Processing betting data...")
        
        # Read historical and current season odds
        try:
            master_odds = pd.read_csv(os.path.join(self.RAW_DIR, 'betting', 'sbr_odds_master.csv'))
            current_odds = pd.read_csv(os.path.join(self.RAW_DIR, 'betting', 'nba_sbr_odds_2025.csv'))
            odds_df = pd.concat([master_odds, current_odds], ignore_index=True)
        except FileNotFoundError as e:
            logger.error(f"Error reading betting files: {e}")
            raise
        
        # Required columns for validation
        required_cols = [
            'GAME_ID', 'GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM', 
            'AWAY_SCORE', 'HOME_SCORE',
            'OPENING_SPREAD', 'CLOSING_SPREAD',
            'AWAY_MONEYLINE_ODDS', 'HOME_MONEYLINE_ODDS',
            'OPENING_TOTAL', 'CLOSING_TOTAL'
        ]
        
        # Validate required columns
        missing_cols = [col for col in required_cols if col not in odds_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in betting data: {missing_cols}")
        
        # Convert date to datetime
        odds_df['GAME_DATE'] = pd.to_datetime(odds_df['GAME_DATE'])
        
        # Sort by date
        odds_df = odds_df.sort_values('GAME_DATE')
        
        # Calculate derived metrics
        odds_df['POINT_DIFF'] = odds_df['AWAY_SCORE'] - odds_df['HOME_SCORE']
        odds_df['TOTAL_POINTS'] = odds_df['AWAY_SCORE'] + odds_df['HOME_SCORE']
        
        # Calculate betting accuracy metrics
        odds_df['ATS_RESULT'] = odds_df.apply(
            lambda x: 1 if x['POINT_DIFF'] > abs(x['CLOSING_SPREAD']) 
            else (-1 if x['POINT_DIFF'] < -abs(x['CLOSING_SPREAD']) 
            else 0), axis=1
        )
        
        odds_df['OVER_UNDER_RESULT'] = odds_df.apply(
            lambda x: 1 if x['TOTAL_POINTS'] > x['CLOSING_TOTAL']
            else (-1 if x['TOTAL_POINTS'] < x['CLOSING_TOTAL']
            else 0), axis=1
        )
        
        # Calculate implied probabilities from moneylines
        def ml_to_prob(ml):
            if pd.isna(ml):
                return np.nan
            if ml > 0:
                return 100 / (ml + 100)
            else:
                return (-ml) / (-ml + 100)
        
        odds_df['AWAY_IMPLIED_PROB'] = odds_df['AWAY_MONEYLINE_ODDS'].apply(ml_to_prob)
        odds_df['HOME_IMPLIED_PROB'] = odds_df['HOME_MONEYLINE_ODDS'].apply(ml_to_prob)
        
        # Calculate market efficiency metrics
        odds_df['ACTUAL_AWAY_WIN'] = (odds_df['POINT_DIFF'] > 0).astype(int)
        odds_df['PREDICTION_ERROR'] = odds_df['AWAY_IMPLIED_PROB'] - odds_df['ACTUAL_AWAY_WIN']
        
        # Validate processed data
        self._validate_betting_data(odds_df)
        
        logger.info("Betting data processing complete")
        return odds_df

    def _validate_betting_data(self, df: pd.DataFrame):
        """Validate processed betting data.
        
        Args:
            df (pd.DataFrame): Processed betting dataframe to validate
        
        Raises:
            ValueError: If validation fails
        """
        # Check for nulls in key columns
        key_cols = ['GAME_ID', 'GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM', 
                    'AWAY_SCORE', 'HOME_SCORE', 'POINT_DIFF', 
                    'TOTAL_POINTS']
        
        null_counts = df[key_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found in betting data: {null_counts[null_counts > 0]}")
        
        # Validate calculation correctness
        if not np.allclose(
            df['POINT_DIFF'],
            df['AWAY_SCORE'] - df['HOME_SCORE'],
            equal_nan=True
        ):
            raise ValueError("Point differential calculation error")
        
        if not np.allclose(
            df['TOTAL_POINTS'],
            df['AWAY_SCORE'] + df['HOME_SCORE'],
            equal_nan=True
        ):
            raise ValueError("Total points calculation error")
        
        # Validate probability ranges
        prob_cols = ['AWAY_IMPLIED_PROB', 'HOME_IMPLIED_PROB']
        for col in prob_cols:
            if ((df[col].dropna() < 0) | (df[col].dropna() > 1)).any():
                raise ValueError(f"Invalid probability values in {col}")
        
        # Validate betting results
        result_cols = ['ATS_RESULT', 'OVER_UNDER_RESULT']
        for col in result_cols:
            if not df[col].isin([-1, 0, 1]).all():
                raise ValueError(f"Invalid betting results in {col}")
            
        logger.info("Betting data validation passed")

    def process_matchup_data(self) -> pd.DataFrame:
        """Process all matchup statistics data.
        
        Returns:
            pd.DataFrame: Processed matchup statistics
        """
        logger.info("Processing matchup statistics...")
        
        # Read and merge different types of matchup stats
        matchup_types = ['boxscorematchupsv3', 'matchupsrollup']
        
        dfs = []
        for matchup_type in matchup_types:
            df = self._read_matchup_data(matchup_type)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            logger.error("No matchup statistics data found")
            return pd.DataFrame()
        
        # Merge all matchup stats
        matchup_stats = pd.concat(dfs, ignore_index=True)
        
        # Process and clean the data
        matchup_stats['game_date'] = pd.to_datetime(matchup_stats['game_date'])
        matchup_stats = matchup_stats.sort_values(['team_id', 'game_date'])
        
        # Calculate matchup metrics
        matchup_stats['matchup_efficiency'] = (
            matchup_stats['points'] / 
            matchup_stats['possessions']
        )
        
        # Calculate rolling statistics
        stats_to_roll = [
            'matchup_efficiency', 'points',
            'possessions', 'assists',
            'turnovers', 'rebounds'
        ]
        
        rolling_stats = self._calculate_rolling_stats(
            matchup_stats,
            stats_to_roll,
            [3, 5, 10]
        )
        
        # Merge rolling stats
        matchup_stats = pd.merge(
            matchup_stats,
            rolling_stats,
            on=['team_id', 'game_date'],
            how='left'
        )
        
        # Validate the processed data
        self._validate_matchup_stats(matchup_stats)
        
        logger.info("Matchup statistics processing complete")
        return matchup_stats
        
    def process_defense_data(self) -> pd.DataFrame:
        """Process all defensive statistics data.
        
        Returns:
            pd.DataFrame: Processed defensive statistics
        """
        logger.info("Processing defensive statistics...")
        
        # Read and merge different types of defense stats
        defense_types = [
            ('boxscoredefensivev2', None),
            ('hustlestatsboxscore', 'team'),
            ('hustlestatsboxscore', 'player'),
            ('leaguehustlestatsteam', None),
            ('leaguehustlestatsplayer', None)
        ]
        
        dfs = []
        for def_type, subfolder in defense_types:
            df = self._read_defense_data(def_type, subfolder)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            logger.error("No defensive statistics data found")
            return pd.DataFrame()
        
        # Merge all defense stats
        defense_stats = pd.concat(dfs, ignore_index=True)
        
        # Process and clean the data
        defense_stats['game_date'] = pd.to_datetime(defense_stats['game_date'])
        defense_stats = defense_stats.sort_values(['team_id', 'game_date'])
        
        # Calculate defensive metrics
        defense_stats['defensive_efficiency'] = (
            defense_stats['opponent_points'] / 
            defense_stats['possessions']
        )
        
        # Calculate rolling statistics
        stats_to_roll = [
            'defensive_efficiency', 'deflections',
            'charges_drawn', 'contested_shots',
            'contested_2pt', 'contested_3pt'
        ]
        
        rolling_stats = self._calculate_rolling_stats(
            defense_stats,
            stats_to_roll,
            [3, 5, 10]
        )
        
        # Merge rolling stats
        defense_stats = pd.merge(
            defense_stats,
            rolling_stats,
            on=['team_id', 'game_date'],
            how='left'
        )
        
        # Validate the processed data
        self._validate_defense_stats(defense_stats)
        
        logger.info("Defensive statistics processing complete")
        return defense_stats
        
    def process_tracking_data(self) -> pd.DataFrame:
        """Process all tracking statistics data.
        
        Returns:
            pd.DataFrame: Processed tracking statistics
        """
        logger.info("Processing tracking statistics...")
        
        # Read and merge different types of tracking stats
        tracking_types = [
            'playergamelogs',
            'shotchartdetail',
            'leaguedashteamstats',
            'leaguedashplayerstats',
            'leaguedashptdefend',
            'leaguedashteamclutch',
            'teamshootingsplits'
        ]
        
        dfs = []
        for track_type in tracking_types:
            df = self._read_tracking_data(track_type)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            logger.error("No tracking statistics data found")
            return pd.DataFrame()
        
        # Merge all tracking stats
        tracking_stats = pd.concat(dfs, ignore_index=True)
        
        # Process and clean the data
        tracking_stats['game_date'] = pd.to_datetime(tracking_stats['game_date'])
        tracking_stats = tracking_stats.sort_values(['team_id', 'game_date'])
        
        # Calculate tracking metrics
        tracking_stats['player_speed'] = (
            tracking_stats['distance_miles'] / 
            tracking_stats['minutes']
        )
        
        # Calculate rolling statistics
        stats_to_roll = [
            'player_speed', 'distance_miles',
            'avg_speed', 'avg_speed_off',
            'avg_speed_def'
        ]
        
        rolling_stats = self._calculate_rolling_stats(
            tracking_stats,
            stats_to_roll,
            [3, 5, 10]
        )
        
        # Merge rolling stats
        tracking_stats = pd.merge(
            tracking_stats,
            rolling_stats,
            on=['team_id', 'game_date'],
            how='left'
        )
        
        # Validate the processed data
        self._validate_tracking_stats(tracking_stats)
        
        logger.info("Tracking statistics processing complete")
        return tracking_stats
        
    def create_features(self, *dataframes: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from processed data.
        
        Args:
            *dataframes: Variable number of processed dataframes to combine
            
        Returns:
            pd.DataFrame: Combined dataset with engineered features
        """
        logger.info("Creating engineered features...")
        
        # Unpack the dataframes
        core_stats, advanced_stats, betting_data, matchup_data, defense_data, tracking_data = dataframes
        
        # Start with core stats as base
        features = core_stats.copy()
        
        # 1. Core Performance Features
        logger.info("Creating core performance features...")
        
        # Rolling performance metrics
        for window in [3, 5, 10]:
            # Team performance
            features[f'win_pct_{window}g'] = features.groupby('team_id')['win'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            features[f'point_diff_{window}g'] = features.groupby('team_id')['point_differential'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Offensive metrics
            features[f'pts_{window}g'] = features.groupby('team_id')['pts'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            features[f'off_rating_{window}g'] = features.groupby('team_id')['offensive_rating'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Defensive metrics
            features[f'def_rating_{window}g'] = features.groupby('team_id')['defensive_rating'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            features[f'net_rating_{window}g'] = features[f'off_rating_{window}g'] - features[f'def_rating_{window}g']
        
        # 2. Player Features
        logger.info("Adding player-specific features...")
        
        # Read player data
        player_stats = pd.read_csv(os.path.join(self.RAW_DIR, 'core', 'nba_player_box_scores.csv'))
        player_advanced = pd.read_csv(os.path.join(self.RAW_DIR, 'advanced', 'player_advanced_stats.csv'))
        
        # Process player stats
        player_stats['game_date'] = pd.to_datetime(player_stats['game_date'])
        player_advanced['game_date'] = pd.to_datetime(player_advanced['game_date'])
        
        # Calculate player performance metrics
        player_stats['player_efficiency'] = (
            player_stats['pts'] + 
            player_stats['reb'] + 
            player_stats['ast'] + 
            player_stats['stl'] + 
            player_stats['blk'] - 
            player_stats['tov']
        ) / player_stats['min']
        
        # Calculate player usage
        player_stats['usage_rate'] = (
            player_stats['fga'] + 
            0.44 * player_stats['fta'] + 
            player_stats['tov']
        ) / player_stats['team_possessions']
        
        # Calculate player impact metrics
        player_stats['player_impact'] = (
            player_stats['plus_minus'] * 
            player_stats['min'] / 48
        )
        
        # Calculate player shooting efficiency
        player_stats['player_ts_pct'] = (
            player_stats['pts'] / 
            (2 * (player_stats['fga'] + 0.44 * player_stats['fta']))
        )
        
        # Calculate player defensive metrics
        player_stats['defensive_impact'] = (
            player_stats['stl'] + 
            player_stats['blk'] + 
            player_stats['def_reb']
        ) / player_stats['min']
        
        # Aggregate player stats by team and game
        team_player_stats = player_stats.groupby(['team_id', 'game_date']).agg({
            'player_efficiency': ['mean', 'max'],
            'usage_rate': ['mean', 'max'],
            'player_impact': ['mean', 'max'],
            'player_ts_pct': ['mean', 'max'],
            'defensive_impact': ['mean', 'max'],
            'min': ['sum', 'max']
        }).reset_index()
        
        # Flatten column names
        team_player_stats.columns = ['_'.join(col).strip('_') for col in team_player_stats.columns.values]
        
        # Merge player stats with features
        features = pd.merge(
            features,
            team_player_stats,
            on=['team_id', 'game_date'],
            how='left'
        )
        
        # Calculate team depth metrics
        features['team_depth'] = features['min_sum'] / 240  # 5 players * 48 minutes
        features['star_power'] = features['player_impact_max'] * features['usage_rate_max']
        
        # Calculate team balance metrics
        features['team_balance'] = (
            features['player_efficiency_mean'] / 
            features['player_efficiency_max']
        )
        
        # Calculate team consistency metrics
        features['team_consistency'] = (
            features['defensive_impact_mean'] * 
            features['player_ts_pct_mean']
        )
        
        # 3. Advanced Metrics
        logger.info("Adding advanced metrics...")
        
        # Merge advanced stats
        features = pd.merge(
            features,
            advanced_stats,
            on=['team_id', 'game_date'],
            how='left',
            suffixes=('', '_advanced')
        )
        
        # Four factors
        features['efg_pct'] = (features['fgm'] + 0.5 * features['fg3m']) / features['fga']
        features['tov_pct'] = features['tov'] / (features['fga'] + 0.44 * features['fta'] + features['tov'])
        features['orb_pct'] = features['oreb'] / (features['oreb'] + features['opp_dreb'])
        features['ftr'] = features['fta'] / features['fga']
        
        # Pace and efficiency
        features['pace'] = features['possessions'] / (features['min'] / 48)
        features['true_shooting_pct'] = features['pts'] / (2 * (features['fga'] + 0.44 * features['fta']))
        
        # 4. Matchup Features
        logger.info("Adding matchup features...")
        
        # Merge matchup data
        features = pd.merge(
            features,
            matchup_data,
            on=['team_id', 'game_date', 'opponent_id'],
            how='left',
            suffixes=('', '_matchup')
        )
        
        # Head-to-head metrics
        features['h2h_win_pct'] = features.groupby(['team_id', 'opponent_id'])['win'].transform('mean')
        features['h2h_point_diff'] = features.groupby(['team_id', 'opponent_id'])['point_differential'].transform('mean')
        
        # 5. Defense Features
        logger.info("Adding defensive features...")
        
        # Merge defense data
        features = pd.merge(
            features,
            defense_data,
            on=['team_id', 'game_date'],
            how='left',
            suffixes=('', '_defense')
        )
        
        # Defensive pressure metrics
        features['defensive_pressure'] = (
            features['deflections'] + 
            features['charges_drawn'] + 
            features['contested_shots']
        ) / features['possessions']
        
        # 6. Betting Features
        logger.info("Adding betting features...")
        
        # Merge betting data
        features = pd.merge(
            features,
            betting_data,
            on=['game_id', 'game_date'],
            how='left',
            suffixes=('', '_betting')
        )
        
        # Market efficiency metrics
        features['line_movement'] = features['closing_spread'] - features['opening_spread']
        features['market_consensus'] = features.groupby('game_id')['closing_spread'].transform('std')
        
        # 7. Contextual Features
        logger.info("Adding contextual features...")
        
        # Rest days
        features['days_rest'] = features.groupby('team_id')['game_date'].diff().dt.days
        features['opponent_days_rest'] = features.groupby('opponent_id')['game_date'].diff().dt.days
        
        # Back-to-back indicator
        features['is_back_to_back'] = (features['days_rest'] == 1).astype(int)
        features['opponent_is_back_to_back'] = (features['opponent_days_rest'] == 1).astype(int)
        
        # Home/away performance
        features['home_win_pct'] = features.groupby(['team_id', 'is_home'])['win'].transform('mean')
        features['away_win_pct'] = features.groupby(['team_id', 'is_home'])['win'].transform('mean')
        
        # 8. Composite Features
        logger.info("Creating composite features...")
        
        # Team strength
        features['team_strength'] = (
            features['net_rating_10g'] * 
            features['win_pct_10g'] * 
            features['point_diff_10g']
        )
        
        # Momentum
        features['momentum'] = (
            features['win_pct_5g'] - 
            features['win_pct_10g']
        )
        
        # Matchup advantage
        features['matchup_advantage'] = (
            features['h2h_win_pct'] * 
            features['h2h_point_diff']
        )
        
        # Market confidence
        features['market_confidence'] = (
            features['market_consensus'] * 
            (1 - abs(features['line_movement']))
        )
        
        # 9. Target Variables
        logger.info("Creating target variables...")
        
        # Game outcome
        features['target_win'] = features['win']
        
        # Point spread
        features['target_spread'] = features['point_differential']
        
        # Total points
        features['target_total'] = features['pts'] + features['opp_pts']
        
        # Clean up
        logger.info("Cleaning up features...")
        
        # Drop unnecessary columns
        drop_cols = [
            'game_id', 'team_abbreviation', 'opponent_abbreviation',
            'wl', 'min', 'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta',
            'oreb', 'dreb', 'ast', 'stl', 'blk', 'tov', 'pf', 'plus_minus',
            'min_sum', 'min_max'  # Add player-specific columns to drop
        ]
        features = features.drop(columns=[col for col in drop_cols if col in features.columns])
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Validate the feature set
        if not self.validate_data(features, "features"):
            logger.error("Feature validation failed")
            raise ValueError("Feature validation failed")
            
        logger.info("Feature engineering completed")
        return features
        
    def validate_data(self, data: pd.DataFrame, data_type: str) -> bool:
        """Validate processed data against defined rules."""
        logger.info(f"Validating {data_type} data...")
        
        # Basic validation checks
        if data.empty:
            logger.error(f"{data_type} data is empty")
            return False
            
        # Check for required columns
        required_cols = self.validation_rules.get(data_type, {}).get('required_columns', [])
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing required columns in {data_type}: {missing_cols}")
            return False
            
        # Check for null values
        null_counts = data.isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found in {data_type}: {null_counts[null_counts > 0]}")
            
        return True
        
    def save_processed_data(self, data: pd.DataFrame, filename: str):
        """Save processed data to the processed directory."""
        output_path = self.PROCESSED_DIR / filename
        data.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
    def run_pipeline(self):
        """Run the complete data processing pipeline."""
        logger.info("Starting data processing pipeline...")
        
        # Process each data type
        core_stats = self.process_core_stats()
        advanced_stats = self.process_advanced_stats()
        betting_data = self.process_betting_data()
        matchup_data = self.process_matchup_data()
        defense_data = self.process_defense_data()
        tracking_data = self.process_tracking_data()
        
        # Combine and create features
        combined_data = self.create_features(
            core_stats,
            advanced_stats,
            betting_data,
            matchup_data,
            defense_data,
            tracking_data
        )
        
        # Validate and save
        if self.validate_data(combined_data, "combined"):
            self.save_processed_data(combined_data, "processed_data.parquet")
            logger.info("Data processing pipeline completed successfully")
        else:
            logger.error("Data validation failed")
            
    def _read_advanced_data(self, data_type: str) -> pd.DataFrame:
        """Read and merge advanced stats data from multiple files.
        
        Args:
            data_type: Type of advanced data (e.g., 'boxscoreadvancedv3', 'boxscorefourfactorv3')
            
        Returns:
            pd.DataFrame: Merged advanced stats data
        """
        try:
            logger.info(f"Reading {data_type} data...")
            files = glob.glob(str(self.RAW_DIR / 'advanced' / data_type / '*.csv'))
            if not files:
                logger.warning(f"No files found for {data_type}")
                return pd.DataFrame()
            
            dfs = []
            for file in files:
                try:
                    df = pd.read_csv(file)
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file}: {str(e)}")
                    continue
            
            if not dfs:
                logger.error(f"No valid data found for {data_type}")
                return pd.DataFrame()
            
            merged_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Merged {len(dfs)} files for {data_type}")
            return merged_df
        except Exception as e:
            logger.error(f"Error processing {data_type}: {str(e)}")
            raise

    def _read_defense_data(self, data_type: str, subfolder: str = None) -> pd.DataFrame:
        """Read and merge defense data from multiple files.
        
        Args:
            data_type: Type of defense data (e.g., 'boxscoredefensivev2', 'hustlestatsboxscore')
            subfolder: Optional subfolder (e.g., 'team', 'player')
            
        Returns:
            pd.DataFrame: Merged defense data
        """
        try:
            logger.info(f"Reading {data_type} data...")
            path = self.RAW_DIR / 'defense' / data_type
            if subfolder:
                path = path / subfolder
            
            files = glob.glob(str(path / '*.csv'))
            if not files:
                logger.warning(f"No files found for {data_type}")
                return pd.DataFrame()
            
            dfs = []
            for file in files:
                try:
                    df = pd.read_csv(file)
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file}: {str(e)}")
                    continue
            
            if not dfs:
                logger.error(f"No valid data found for {data_type}")
                return pd.DataFrame()
            
            merged_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Merged {len(dfs)} files for {data_type}")
            return merged_df
        except Exception as e:
            logger.error(f"Error processing {data_type}: {str(e)}")
            raise

    def _read_matchup_data(self, data_type: str) -> pd.DataFrame:
        """Read and merge matchup data from multiple files.
        
        Args:
            data_type: Type of matchup data (e.g., 'boxscorematchupsv3', 'matchupsrollup')
            
        Returns:
            pd.DataFrame: Merged matchup data
        """
        try:
            logger.info(f"Reading {data_type} data...")
            if data_type == 'matchupsrollup':
                # Single file for the season
                return pd.read_csv(self.RAW_DIR / 'matchups' / data_type / 'matchupsrollup.csv')
            else:
                # Multiple files for each game
                files = glob.glob(str(self.RAW_DIR / 'matchups' / data_type / '*.csv'))
                if not files:
                    logger.warning(f"No files found for {data_type}")
                    return pd.DataFrame()
                
                dfs = []
                for file in files:
                    try:
                        df = pd.read_csv(file)
                        dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error reading {file}: {str(e)}")
                        continue
                
                if not dfs:
                    logger.error(f"No valid data found for {data_type}")
                    return pd.DataFrame()
                
                merged_df = pd.concat(dfs, ignore_index=True)
                logger.info(f"Merged {len(dfs)} files for {data_type}")
                return merged_df
        except Exception as e:
            logger.error(f"Error processing {data_type}: {str(e)}")
            raise

    def _read_tracking_data(self, data_type: str) -> pd.DataFrame:
        """Read and merge tracking data from multiple files.
        
        Args:
            data_type: Type of tracking data (e.g., 'playergamelogs', 'shotchartdetail')
            
        Returns:
            pd.DataFrame: Merged tracking data
        """
        try:
            logger.info(f"Reading {data_type} data...")
            if data_type == 'playergamelogs':
                # Multiple files for each player
                files = glob.glob(str(self.RAW_DIR / 'tracking' / data_type / '*.csv'))
                if not files:
                    logger.warning(f"No files found for {data_type}")
                    return pd.DataFrame()
                
                dfs = []
                for file in files:
                    try:
                        df = pd.read_csv(file)
                        dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error reading {file}: {str(e)}")
                        continue
                
                if not dfs:
                    logger.error(f"No valid data found for {data_type}")
                    return pd.DataFrame()
                
                merged_df = pd.concat(dfs, ignore_index=True)
                logger.info(f"Merged {len(dfs)} files for {data_type}")
                return merged_df
            elif data_type == 'shotchartdetail':
                # Multiple files for each game
                files = glob.glob(str(self.RAW_DIR / 'tracking' / data_type / '*.csv'))
                if not files:
                    logger.warning(f"No files found for {data_type}")
                    return pd.DataFrame()
                
                dfs = []
                for file in files:
                    try:
                        df = pd.read_csv(file)
                        dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error reading {file}: {str(e)}")
                        continue
                
                if not dfs:
                    logger.error(f"No valid data found for {data_type}")
                    return pd.DataFrame()
                
                merged_df = pd.concat(dfs, ignore_index=True)
                logger.info(f"Merged {len(dfs)} files for {data_type}")
                return merged_df
            else:
                # Single file for the season
                return pd.read_csv(self.RAW_DIR / 'tracking' / f'{data_type}_2024-25.csv')
        except Exception as e:
            logger.error(f"Error processing {data_type}: {str(e)}")
            raise

    def _validate_advanced_stats(self, df: pd.DataFrame) -> bool:
        """Validate processed advanced statistics.
        
        Args:
            df: DataFrame containing processed advanced statistics
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        logger.info("Validating advanced statistics...")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        try:
            # Get validation rules for advanced stats
            rules = self.validation_rules.get('advanced_stats', {})
            logger.info(f"Validation rules: {rules}")
            
            # Check for empty dataframe
            if df.empty:
                logger.error("Empty dataframe provided for validation")
                return False
                
            # Check required columns
            required_cols = rules.get('required_columns', [])
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in advanced stats: {missing_cols}")
                return False
                
            # Check numeric columns
            numeric_cols = rules.get('numeric_columns', [])
            for col in numeric_cols:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        logger.error(f"Column {col} must be numeric")
                        return False
                    logger.info(f"Column {col} stats - min: {df[col].min()}, max: {df[col].max()}, mean: {df[col].mean():.2f}")
                else:
                    logger.error(f"Missing numeric column: {col}")
                    return False
                    
            # Check value ranges
            value_ranges = rules.get('value_ranges', {})
            for col, ranges in value_ranges.items():
                if col in df.columns:
                    min_val = ranges.get('min')
                    max_val = ranges.get('max')
                    
                    if min_val is not None and df[col].min() < min_val:
                        logger.error(f"Column {col} contains values below minimum {min_val}")
                        return False
                        
                    if max_val is not None and df[col].max() > max_val:
                        logger.error(f"Column {col} contains values above maximum {max_val}")
                        return False
                        
            # Check for null values in required columns
            null_cols = df[required_cols].isnull().sum()
            if null_cols.any():
                logger.error(f"Null values found in columns: {null_cols[null_cols > 0].to_dict()}")
                return False
                
            # Check data types
            if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
                logger.error("game_date must be datetime type")
                return False
                
            # Check for duplicate games
            duplicates = df.duplicated(subset=['team_id', 'game_date'], keep=False)
            if duplicates.any():
                logger.error(f"Found {duplicates.sum()} duplicate games")
                return False
                
            logger.info("Advanced statistics validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error during advanced stats validation: {str(e)}")
            return False

    def _validate_defense_stats(self, df: pd.DataFrame) -> bool:
        """Validate processed defense statistics.
        
        Args:
            df: DataFrame containing processed defense statistics
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        logger.info("Validating defense statistics...")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        try:
            # Get validation rules for defense stats
            rules = self.validation_rules.get('defense_stats', {})
            logger.info(f"Validation rules: {rules}")
            
            # Check for empty dataframe
            if df.empty:
                logger.error("Empty dataframe provided for validation")
                return False
                
            # Check required columns
            required_cols = rules.get('required_columns', [])
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in defense stats: {missing_cols}")
                return False
                
            # Check numeric columns
            numeric_cols = rules.get('numeric_columns', [])
            for col in numeric_cols:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        logger.error(f"Column {col} must be numeric")
                        return False
                    logger.info(f"Column {col} stats - min: {df[col].min()}, max: {df[col].max()}, mean: {df[col].mean():.2f}")
                else:
                    logger.error(f"Missing numeric column: {col}")
                    return False
                    
            # Check value ranges
            value_ranges = rules.get('value_ranges', {})
            for col, ranges in value_ranges.items():
                if col in df.columns:
                    min_val = ranges.get('min')
                    max_val = ranges.get('max')
                    
                    if min_val is not None and df[col].min() < min_val:
                        logger.error(f"Column {col} contains values below minimum {min_val}")
                        return False
                        
                    if max_val is not None and df[col].max() > max_val:
                        logger.error(f"Column {col} contains values above maximum {max_val}")
                        return False
                        
            # Check for null values in required columns
            null_cols = df[required_cols].isnull().sum()
            if null_cols.any():
                logger.error(f"Null values found in columns: {null_cols[null_cols > 0].to_dict()}")
                return False
                
            # Check data types
            if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
                logger.error("game_date must be datetime type")
                return False
                
            # Check for duplicate games
            duplicates = df.duplicated(subset=['team_id', 'game_date'], keep=False)
            if duplicates.any():
                logger.error(f"Found {duplicates.sum()} duplicate games")
                return False
                
            logger.info("Defense statistics validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error during defense stats validation: {str(e)}")
            return False

    def _validate_matchup_stats(self, df: pd.DataFrame) -> bool:
        """Validate processed matchup statistics.
        
        Args:
            df: DataFrame containing processed matchup statistics
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        logger.info("Validating matchup statistics...")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        try:
            # Get validation rules for matchup stats
            rules = self.validation_rules.get('matchup_stats', {})
            logger.info(f"Validation rules: {rules}")
            
            # Check for empty dataframe
            if df.empty:
                logger.error("Empty dataframe provided for validation")
                return False
                
            # Check required columns
            required_cols = rules.get('required_columns', [])
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in matchup stats: {missing_cols}")
                return False
                
            # Check numeric columns
            numeric_cols = rules.get('numeric_columns', [])
            for col in numeric_cols:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        logger.error(f"Column {col} must be numeric")
                        return False
                    logger.info(f"Column {col} stats - min: {df[col].min()}, max: {df[col].max()}, mean: {df[col].mean():.2f}")
                else:
                    logger.error(f"Missing numeric column: {col}")
                    return False
                    
            # Check value ranges
            value_ranges = rules.get('value_ranges', {})
            for col, ranges in value_ranges.items():
                if col in df.columns:
                    min_val = ranges.get('min')
                    max_val = ranges.get('max')
                    
                    if min_val is not None and df[col].min() < min_val:
                        logger.error(f"Column {col} contains values below minimum {min_val}")
                        return False
                        
                    if max_val is not None and df[col].max() > max_val:
                        logger.error(f"Column {col} contains values above maximum {max_val}")
                        return False
                        
            # Check for null values in required columns
            null_cols = df[required_cols].isnull().sum()
            if null_cols.any():
                logger.error(f"Null values found in columns: {null_cols[null_cols > 0].to_dict()}")
                return False
                
            # Check data types
            if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
                logger.error("game_date must be datetime type")
                return False
                
            # Check for duplicate games
            duplicates = df.duplicated(subset=['team_id', 'game_date'], keep=False)
            if duplicates.any():
                logger.error(f"Found {duplicates.sum()} duplicate games")
                return False
                
            logger.info("Matchup statistics validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error during matchup stats validation: {str(e)}")
            return False

    def _validate_tracking_stats(self, df: pd.DataFrame) -> bool:
        """Validate processed tracking statistics.
        
        Args:
            df: DataFrame containing processed tracking statistics
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        logger.info("Validating tracking statistics...")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        try:
            # Get validation rules for tracking stats
            rules = self.validation_rules.get('tracking_stats', {})
            logger.info(f"Validation rules: {rules}")
            
            # Check for empty dataframe
            if df.empty:
                logger.error("Empty dataframe provided for validation")
                return False
                
            # Check required columns
            required_cols = rules.get('required_columns', [])
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in tracking stats: {missing_cols}")
                return False
                
            # Check numeric columns
            numeric_cols = rules.get('numeric_columns', [])
            for col in numeric_cols:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        logger.error(f"Column {col} must be numeric")
                        return False
                    logger.info(f"Column {col} stats - min: {df[col].min()}, max: {df[col].max()}, mean: {df[col].mean():.2f}")
                else:
                    logger.error(f"Missing numeric column: {col}")
                    return False
                    
            # Check value ranges
            value_ranges = rules.get('value_ranges', {})
            for col, ranges in value_ranges.items():
                if col in df.columns:
                    min_val = ranges.get('min')
                    max_val = ranges.get('max')
                    
                    if min_val is not None and df[col].min() < min_val:
                        logger.error(f"Column {col} contains values below minimum {min_val}")
                        return False
                        
                    if max_val is not None and df[col].max() > max_val:
                        logger.error(f"Column {col} contains values above maximum {max_val}")
                        return False
                        
            # Check for null values in required columns
            null_cols = df[required_cols].isnull().sum()
            if null_cols.any():
                logger.error(f"Null values found in columns: {null_cols[null_cols > 0].to_dict()}")
                return False
                
            # Check data types
            if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
                logger.error("game_date must be datetime type")
                return False
                
            # Check for duplicate games
            duplicates = df.duplicated(subset=['team_id', 'game_date'], keep=False)
            if duplicates.any():
                logger.error(f"Found {duplicates.sum()} duplicate games")
                return False
                
            logger.info("Tracking statistics validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error during tracking stats validation: {str(e)}")
            return False

if __name__ == "__main__":
    processor = DataProcessor()
    processor.run_pipeline() 