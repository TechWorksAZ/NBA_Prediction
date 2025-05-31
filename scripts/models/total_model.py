"""
Total Model Trainer

This module handles training models for:
1. Total points predictions
2. Over/Under predictions
"""

from typing import Dict, Tuple, Any, List
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from .model_trainer import NBAModelTraining
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import wandb
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

class TotalModelTrainer(NBAModelTraining):
    """Trainer for total points prediction models."""
    
    def __init__(self, use_wandb: bool = True):
        super().__init__(use_wandb)
        self.exclude_columns = ['GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'TEAM_CITY', 'TOTAL_GAME_MINUTES']
        
        # Update model parameters for better accuracy
        self.model_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': ['mae', 'rmse'],
            'max_depth': 8,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'tree_method': 'hist',
            'random_state': 42
        })
        
        # Add cross-validation parameters
        self.cv_params = {
            'n_splits': 5,
            'shuffle': True,
            'random_state': 42
        }
        
        self.target_columns = {
            'total_points': 'TOTAL_POINTS'  # Total points prediction
        }
        
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select the most predictive features using various methods.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            List of selected feature names
        """
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(X, y)
        mi_features = pd.Series(mi_scores, index=X.columns)
        mi_features = mi_features.sort_values(ascending=False)
        
        # Use Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_features = pd.Series(rf.feature_importances_, index=X.columns)
        rf_features = rf_features.sort_values(ascending=False)
        
        # Select features that are important in either method
        top_mi_features = set(mi_features.head(30).index)
        top_rf_features = set(rf_features.head(30).index)
        selected_features = list(top_mi_features.union(top_rf_features))
        
        # Log feature selection results
        self.logger.info("\nTop 10 features by mutual information:")
        self.logger.info(mi_features.head(10))
        self.logger.info("\nTop 10 features by Random Forest importance:")
        self.logger.info(rf_features.head(10))
        self.logger.info(f"\nSelected {len(selected_features)} features")
        
        return selected_features
    
    def prepare_features(self, features: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for total points prediction.
        
        Args:
            features: Dictionary of feature DataFrames
            
        Returns:
            Tuple of (X, y) where X is the feature DataFrame and y is the target Series
        """
        self.logger.info("Preparing features for total points prediction...")
        
        # Get DataFrames
        team_game = features['team_game'].copy()
        target = features['target'].copy()
        betting = features['betting'].copy()
        
        # Convert GAME_ID to string in all dataframes to ensure consistent type
        for df in [team_game, target, betting]:
            df['GAME_ID'] = df['GAME_ID'].astype(str).str.replace('.0', '')
        
        # === DIAGNOSTICS: Print GAME_ID samples and intersections ===
        print('\n[DIAGNOSTIC] First 10 GAME_IDs:')
        print('team_game:', team_game['GAME_ID'].unique()[:10])
        print('target   :', target['GAME_ID'].unique()[:10])
        print('betting  :', betting['GAME_ID'].unique()[:10])
        team_ids = set(team_game['GAME_ID'].unique())
        target_ids = set(target['GAME_ID'].unique())
        betting_ids = set(betting['GAME_ID'].unique())
        print(f'[DIAGNOSTIC] team_game ∩ target: {len(team_ids & target_ids)}')
        print(f'[DIAGNOSTIC] team_game ∩ betting: {len(team_ids & betting_ids)}')
        print(f'[DIAGNOSTIC] target ∩ betting: {len(target_ids & betting_ids)}')
        print(f'[DIAGNOSTIC] team_game ∩ target ∩ betting: {len(team_ids & target_ids & betting_ids)}')
        # Print a few actual matching IDs if any
        print('[DIAGNOSTIC] Example matching GAME_IDs:', list(team_ids & target_ids & betting_ids)[:5])
        # === END DIAGNOSTICS ===
        
        # Fix mixed type columns
        mixed_type_cols = [
            'TEAM_ID', 'HOME_TEAM_ID', 'AWAY_TEAM_ID',
            'TEAM_NAME', 'TEAM_CITY', 'TEAM_ABBREVIATION'
        ]
        for col in mixed_type_cols:
            if col in team_game.columns and team_game[col].dtype == 'object':
                # Try to convert to numeric, coercing errors to NaN
                team_game[col] = pd.to_numeric(team_game[col], errors='coerce')
        
        # Log initial shapes
        self.logger.info(f"Initial team_game shape: {team_game.shape}")
        self.logger.info(f"Initial target shape: {target.shape}")
        self.logger.info(f"Initial betting shape: {betting.shape}")
        
        # Map team IDs to names
        team_id_to_name = team_game[['TEAM_ID', 'TEAM_NAME', 'TEAM_CITY']].drop_duplicates()
        team_id_to_city = dict(zip(team_id_to_name['TEAM_ID'], team_id_to_name['TEAM_CITY']))
        team_id_to_name_only = dict(zip(team_id_to_name['TEAM_ID'], team_id_to_name['TEAM_NAME']))
        
        team_game['HOME_TEAM_FULL'] = team_game['HOME_TEAM_ID'].map(lambda tid: f"{team_id_to_city.get(tid, '')} {team_id_to_name_only.get(tid, '')}".strip())
        team_game['AWAY_TEAM_FULL'] = team_game['AWAY_TEAM_ID'].map(lambda tid: f"{team_id_to_city.get(tid, '')} {team_id_to_name_only.get(tid, '')}".strip())
        
        # Standardize team names (strip, title)
        team_game['HOME_TEAM_FULL'] = team_game['HOME_TEAM_FULL'].str.strip().str.title()
        team_game['AWAY_TEAM_FULL'] = team_game['AWAY_TEAM_FULL'].str.strip().str.title()
        target['HOME_TEAM'] = target['HOME_TEAM'].str.strip().str.title()
        target['AWAY_TEAM'] = target['AWAY_TEAM'].str.strip().str.title()
        
        # Normalize team names
        team_name_map = {
            'La Clippers': 'Los Angeles Clippers',
            'La Lakers': 'Los Angeles Lakers',
            'Philadelphia 76Ers': 'Philadelphia 76ers',
            'New York Knicks': 'New York Knicks',
            'Boston Celtics': 'Boston Celtics',
            'Minnesota Timberwolves': 'Minnesota Timberwolves',
            'Charlotte Hornets': 'Charlotte Hornets',
            'Houston Rockets': 'Houston Rockets',
            'Chicago Bulls': 'Chicago Bulls',
            'New Orleans Pelicans': 'New Orleans Pelicans',
            'Milwaukee Bucks': 'Milwaukee Bucks',
            'Brooklyn Nets': 'Brooklyn Nets',
            'Miami Heat': 'Miami Heat',
            'Orlando Magic': 'Orlando Magic',
            'Washington Wizards': 'Washington Wizards',
            'Atlanta Hawks': 'Atlanta Hawks',
            'Cleveland Cavaliers': 'Cleveland Cavaliers',
            'Detroit Pistons': 'Detroit Pistons',
            'Indiana Pacers': 'Indiana Pacers',
            'Toronto Raptors': 'Toronto Raptors',
            'Denver Nuggets': 'Denver Nuggets',
            'Oklahoma City Thunder': 'Oklahoma City Thunder',
            'Portland Trail Blazers': 'Portland Trail Blazers',
            'Utah Jazz': 'Utah Jazz',
            'Dallas Mavericks': 'Dallas Mavericks',
            'Memphis Grizzlies': 'Memphis Grizzlies',
            'Phoenix Suns': 'Phoenix Suns',
            'Sacramento Kings': 'Sacramento Kings',
            'San Antonio Spurs': 'San Antonio Spurs',
            'Golden State Warriors': 'Golden State Warriors'
        }
        def normalize_team_name(name):
            if pd.isna(name):
                return name
            return team_name_map.get(name, name)
        team_game['HOME_TEAM_FULL'] = team_game['HOME_TEAM_FULL'].apply(normalize_team_name)
        team_game['AWAY_TEAM_FULL'] = team_game['AWAY_TEAM_FULL'].apply(normalize_team_name)
        target['HOME_TEAM'] = target['HOME_TEAM'].apply(normalize_team_name)
        target['AWAY_TEAM'] = target['AWAY_TEAM'].apply(normalize_team_name)
        
        # Filter out 'Nan Nan' and empty string rows
        team_game = team_game[(team_game['HOME_TEAM_FULL'] != 'Nan Nan') & (team_game['AWAY_TEAM_FULL'] != 'Nan Nan')]
        team_game = team_game[(team_game['HOME_TEAM_FULL'] != '') & (team_game['AWAY_TEAM_FULL'] != '')]
        
        # Log unmapped team IDs
        unmapped_home_ids = team_game.loc[team_game['HOME_TEAM_FULL'] == '', 'HOME_TEAM_ID'].unique().tolist()
        unmapped_away_ids = team_game.loc[team_game['AWAY_TEAM_FULL'] == '', 'AWAY_TEAM_ID'].unique().tolist()
        if unmapped_home_ids:
            self.logger.warning(f"Unmapped HOME_TEAM_IDs: {unmapped_home_ids}")
        if unmapped_away_ids:
            self.logger.warning(f"Unmapped AWAY_TEAM_IDs: {unmapped_away_ids}")
        
        # Log shapes after each merge
        self.logger.info(f"\nShape after team name standardization: {team_game.shape}")
        self.logger.info(f"Shape after filtering non-NBA teams: {team_game.shape}")
        
        # Diagnostic logging before merging
        self.logger.info('Sample team_game rows:')
        self.logger.info(team_game.head(5).to_string())
        self.logger.info('Sample betting rows:')
        self.logger.info(betting.head(5).to_string())
        self.logger.info('Sample target rows:')
        self.logger.info(target.head(5).to_string())
        
        # First merge with target to get game-level information
        df = pd.merge(
            team_game,
            target[['GAME_ID', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_SCORE', 'AWAY_SCORE']],
            on='GAME_ID',
            how='inner'
        )
        self.logger.info(f"Shape after merging with target: {df.shape}")
        
        # Then merge with betting data
        df = pd.merge(
            df,
            betting[['GAME_ID', 'HOME_SCORE', 'AWAY_SCORE']],
            on='GAME_ID',
            how='inner',
            suffixes=('', '_betting')
        )
        self.logger.info(f"Shape after merging with betting: {df.shape}")
        
        if df.shape[0] == 0:
            self.logger.error('Merged DataFrame is empty! Logging diagnostics:')
            self.logger.error('team_game unique GAME_IDs:')
            self.logger.error(team_game['GAME_ID'].unique()[:10])
            self.logger.error('betting unique GAME_IDs:')
            self.logger.error(betting['GAME_ID'].unique()[:10])
            self.logger.error('target unique GAME_IDs:')
            self.logger.error(target['GAME_ID'].unique()[:10])
            self.logger.error('Sample team_game rows:')
            self.logger.error(team_game.head(10).to_string())
            self.logger.error('Sample betting rows:')
            self.logger.error(betting.head(10).to_string())
            self.logger.error('Sample target rows:')
            self.logger.error(target.head(10).to_string())
            raise ValueError('Merged DataFrame is empty after merging on GAME_ID! Check logs for diagnostics.')
        
        # Calculate opponent points
        df['OPP_POINTS'] = df.apply(
            lambda row: row['AWAY_SCORE'] if row['TEAM_ID'] == row['HOME_TEAM_ID'] else row['HOME_SCORE'],
            axis=1
        )
        
        # Calculate total points
        df['TOTAL_POINTS'] = df['HOME_SCORE'] + df['AWAY_SCORE']
        self.logger.info(f"Missing values in TOTAL_POINTS: {df['TOTAL_POINTS'].isna().sum()}")
        self.logger.info(f"Total points range: {df['TOTAL_POINTS'].min()} - {df['TOTAL_POINTS'].max()}")
        self.logger.info(f"Average total points: {df['TOTAL_POINTS'].mean():.2f}")
        
        # Add data validation
        def validate_data(df: pd.DataFrame, stage: str) -> None:
            """Validate data at different stages of processing.
            
            Args:
                df: DataFrame to validate
                stage: Current processing stage
            """
            self.logger.info(f"\nValidating data at {stage} stage:")
            
            # Check for required columns
            required_cols = ['GAME_ID', 'HOME_TEAM_FULL', 'AWAY_TEAM_FULL', 'HOME_SCORE', 'AWAY_SCORE', 'TOTAL_POINTS']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns at {stage}: {missing_cols}")
            
            # Check for duplicate games
            dupes = df['GAME_ID'].duplicated().sum()
            if dupes > 0:
                self.logger.warning(f"Found {dupes} duplicate GAME_IDs at {stage}")
            
            # Check for invalid scores
            invalid_scores = df[
                (df['HOME_SCORE'] < 0) | 
                (df['AWAY_SCORE'] < 0) |
                (df['HOME_SCORE'].isna()) |
                (df['AWAY_SCORE'].isna())
            ]
            if len(invalid_scores) > 0:
                self.logger.warning(f"Found {len(invalid_scores)} invalid scores at {stage}")
                self.logger.warning(f"Invalid score examples:\n{invalid_scores[['GAME_ID', 'HOME_SCORE', 'AWAY_SCORE']].head()}")
            
            # Check for missing team names
            missing_teams = df[
                (df['HOME_TEAM_FULL'].isna()) | 
                (df['AWAY_TEAM_FULL'].isna()) |
                (df['HOME_TEAM_FULL'] == '') |
                (df['AWAY_TEAM_FULL'] == '')
            ]
            if len(missing_teams) > 0:
                self.logger.warning(f"Found {len(missing_teams)} missing team names at {stage}")
                self.logger.warning(f"Missing team examples:\n{missing_teams[['GAME_ID', 'HOME_TEAM_FULL', 'AWAY_TEAM_FULL']].head()}")
            
            # Check for missing values in key columns
            key_cols = ['POSSESSIONS', 'PACE', 'OFF_RTG', 'DEF_RTG', 'TOTAL_POINTS']
            for col in key_cols:
                if col in df.columns:
                    missing = df[col].isna().sum()
                    if missing > 0:
                        self.logger.warning(f"Found {missing} missing values in {col}")
            
            # Log data quality metrics
            self.logger.info("\nData quality metrics:")
            self.logger.info(f"Total rows: {len(df)}")
            self.logger.info(f"Unique games: {df['GAME_ID'].nunique()}")
            self.logger.info(f"Unique teams: {df['TEAM_ID'].nunique()}")
            if 'TOTAL_POINTS' in df.columns:
                self.logger.info(f"Total points range: {df['TOTAL_POINTS'].min():.1f} - {df['TOTAL_POINTS'].max():.1f}")
                self.logger.info(f"Average total points: {df['TOTAL_POINTS'].mean():.1f}")
        
        # Validate data at each stage
        validate_data(df, "after merging")
        
        # Add feature engineering
        def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
            """Create additional features for total points prediction.
            
            Args:
                df: Input DataFrame
                
            Returns:
                DataFrame with additional features
            """
            # Calculate pace-related features
            df['POSSESSIONS'] = df['FIELD_GOALS_ATTEMPTED'] - df['REBOUNDS_OFFENSIVE'] + df['TURNOVERS'] + 0.44 * df['FREE_THROWS_ATTEMPTED']
            df['PACE'] = df['POSSESSIONS'] / (df['TOTAL_GAME_MINUTES'] / 48)
            
            # Calculate shooting efficiency
            df['EFG_PCT'] = (df['FIELD_GOALS_MADE'] + 0.5 * df['THREE_POINTERS_MADE']) / df['FIELD_GOALS_ATTEMPTED']
            df['TS_PCT'] = df['POINTS'] / (2 * (df['FIELD_GOALS_ATTEMPTED'] + 0.44 * df['FREE_THROWS_ATTEMPTED']))
            df['FG3_RATE'] = df['THREE_POINTERS_ATTEMPTED'] / df['FIELD_GOALS_ATTEMPTED']
            df['FT_RATE'] = df['FREE_THROWS_ATTEMPTED'] / df['FIELD_GOALS_ATTEMPTED']
            
            # Calculate offensive and defensive ratings
            df['OFF_RTG'] = df['POINTS'] / df['POSSESSIONS'] * 100
            df['DEF_RTG'] = df['OPP_POINTS'] / df['POSSESSIONS'] * 100
            df['NET_RTG'] = df['OFF_RTG'] - df['DEF_RTG']
            
            # Calculate turnover and rebound rates
            df['TOV_RATE'] = df['TURNOVERS'] / df['POSSESSIONS']
            df['OREB_RATE'] = df['REBOUNDS_OFFENSIVE'] / (df['REBOUNDS_OFFENSIVE'] + df['OPP_REBOUNDS_DEFENSIVE'])
            df['DREB_RATE'] = df['REBOUNDS_DEFENSIVE'] / (df['REBOUNDS_DEFENSIVE'] + df['OPP_REBOUNDS_OFFENSIVE'])
            df['TREB_RATE'] = (df['REBOUNDS_OFFENSIVE'] + df['REBOUNDS_DEFENSIVE']) / (df['REBOUNDS_TOTAL'] + df['OPP_REBOUNDS_TOTAL'])
            
            # Calculate assist and steal rates
            df['AST_RATE'] = df['ASSISTS'] / df['FIELD_GOALS_MADE']
            df['STL_RATE'] = df['STEALS'] / df['POSSESSIONS']
            df['BLK_RATE'] = df['BLOCKS'] / df['POSSESSIONS']
            
            # Calculate home/away advantage features
            df['HOME_ADVANTAGE'] = df['HOME_TEAM_FULL'].notna().astype(int)
            df['REST_DAYS'] = df['DAYS_REST'].fillna(0)
            df['BACK_TO_BACK'] = (df['REST_DAYS'] == 0).astype(int)
            
            # Calculate team strength metrics
            df['TEAM_STRENGTH'] = df['OFF_RTG'] * 0.7 + df['DEF_RTG'] * 0.3
            df['OPP_STRENGTH'] = df['OPP_OFF_RTG'] * 0.7 + df['OPP_DEF_RTG'] * 0.3
            df['STRENGTH_DIFF'] = df['TEAM_STRENGTH'] - df['OPP_STRENGTH']
            
            # Calculate rolling averages (3, 5, and 10 games)
            for window in [3, 5, 10]:
                for col in ['POINTS', 'POSSESSIONS', 'PACE', 'EFG_PCT', 'TS_PCT', 'OFF_RTG', 'DEF_RTG', 
                           'TOV_RATE', 'OREB_RATE', 'DREB_RATE', 'AST_RATE', 'STL_RATE', 'BLK_RATE']:
                    df[f'{col}_ROLLING_{window}'] = df.groupby('TEAM_ID')[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    
                    # Add rolling standard deviation
                    df[f'{col}_ROLLING_{window}_STD'] = df.groupby('TEAM_ID')[col].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
            
            # Calculate momentum features
            df['WIN_STREAK'] = df.groupby('TEAM_ID')['WL'].transform(
                lambda x: x.rolling(10, min_periods=1).sum()
            )
            df['LOSS_STREAK'] = df.groupby('TEAM_ID')['WL'].transform(
                lambda x: (1 - x).rolling(10, min_periods=1).sum()
            )
            
            # Calculate matchup-specific features
            df['PACE_MATCHUP'] = df['PACE'] * df['OPP_PACE']
            df['OFF_DEF_MATCHUP'] = df['OFF_RTG'] * df['OPP_DEF_RTG']
            df['DEF_OFF_MATCHUP'] = df['DEF_RTG'] * df['OPP_OFF_RTG']
            
            # Calculate season-long trends
            df['SEASON_GAMES'] = df.groupby('TEAM_ID').cumcount() + 1
            df['SEASON_PACE_TREND'] = df.groupby('TEAM_ID')['PACE'].transform(
                lambda x: x.rolling(20, min_periods=1).mean()
            )
            df['SEASON_OFF_TREND'] = df.groupby('TEAM_ID')['OFF_RTG'].transform(
                lambda x: x.rolling(20, min_periods=1).mean()
            )
            df['SEASON_DEF_TREND'] = df.groupby('TEAM_ID')['DEF_RTG'].transform(
                lambda x: x.rolling(20, min_periods=1).mean()
            )
            
            # Fill missing values with appropriate defaults
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            return df
        
        # Apply feature engineering
        df = engineer_features(df)
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in [
            'GAME_ID', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_TEAM_FULL', 'AWAY_TEAM_FULL', 
            'HOME_SCORE', 'AWAY_SCORE', 'TOTAL_POINTS'
        ]]
        
        # Filter out rows with NaN in target
        df = df.dropna(subset=['TOTAL_POINTS'])
        
        # Split into X and y
        X = df[feature_cols]
        y = df['TOTAL_POINTS']
        
        # Select most predictive features
        selected_features = self.select_features(X, y)
        X = X[selected_features]
        
        self.logger.info(f"Final X shape: {X.shape}")
        self.logger.info(f"Final y shape: {y.shape}")
        
        return X, y
    
    def train_model(self, target_name: str, target_column: str, features: Dict[str, pd.DataFrame]) -> Any:
        """Train a model for the specified target.
        
        Args:
            target_name: Name of the target
            target_column: Column name for the target
            features: Dictionary of feature DataFrames
            
        Returns:
            Trained model
        """
        try:
            # Prepare features
            X, y = self.prepare_features(features)
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No data available for training after feature preparation")
            
            # Split data into train and validation sets
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Set up early stopping
            evals = [(dtrain, 'train'), (dval, 'val')]
            early_stopping_rounds = 50
            
            # Train model with cross-validation
            from sklearn.model_selection import KFold
            kf = KFold(**self.cv_params)
            
            cv_results = []
            best_model = None
            best_score = float('inf')
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                try:
                    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    dtrain_fold = xgb.DMatrix(X_fold_train, label=y_fold_train)
                    dval_fold = xgb.DMatrix(X_fold_val, label=y_fold_val)
                    
                    model = xgb.train(
                        self.model_params,
                        dtrain_fold,
                        num_boost_round=self.model_params['n_estimators'],
                        evals=[(dtrain_fold, 'train'), (dval_fold, 'val')],
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=100
                    )
                    
                    # Evaluate fold
                    val_pred = model.predict(dval_fold)
                    mae = mean_absolute_error(y_fold_val, val_pred)
                    rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
                    cv_results.append({'fold': fold, 'mae': mae, 'rmse': rmse})
                    
                    # Track best model
                    if rmse < best_score:
                        best_score = rmse
                        best_model = model
                    
                    self.logger.info(f"Fold {fold + 1} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"Error in fold {fold + 1}: {str(e)}")
                    continue
            
            if not cv_results:
                raise ValueError("No successful cross-validation folds")
            
            # Train final model on full dataset
            final_model = xgb.train(
                self.model_params,
                dtrain,
                num_boost_round=self.model_params['n_estimators'],
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=100
            )
            
            # Log feature importance
            importance = final_model.get_score(importance_type='gain')
            importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            self.logger.info("\nTop 10 most important features:")
            for feat, imp in importance[:10]:
                self.logger.info(f"{feat}: {imp:.4f}")
            
            # Log cross-validation results
            avg_mae = np.mean([r['mae'] for r in cv_results])
            avg_rmse = np.mean([r['rmse'] for r in cv_results])
            std_mae = np.std([r['mae'] for r in cv_results])
            std_rmse = np.std([r['rmse'] for r in cv_results])
            
            self.logger.info(f"\nCross-validation results:")
            self.logger.info(f"Average MAE: {avg_mae:.2f} ± {std_mae:.2f}")
            self.logger.info(f"Average RMSE: {avg_rmse:.2f} ± {std_rmse:.2f}")
            
            if self.use_wandb:
                wandb.log({
                    f"{target_name}_cv_mae": avg_mae,
                    f"{target_name}_cv_mae_std": std_mae,
                    f"{target_name}_cv_rmse": avg_rmse,
                    f"{target_name}_cv_rmse_std": std_rmse,
                    f"{target_name}_feature_importance": wandb.Table(
                        data=[[feat, imp] for feat, imp in importance[:20]],
                        columns=["feature", "importance"]
                    )
                })
            
            return final_model
            
        except Exception as e:
            self.logger.error(f"Error training {target_name} model: {str(e)}")
            raise
    
    def train(self, features: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train total points prediction models.
        
        Args:
            features: Dictionary of feature DataFrames
            
        Returns:
            Dictionary of trained models and metrics
        """
        logger.info("Training total points prediction models...")
        
        results = {}
        for target_name, target_column in self.target_columns.items():
            try:
                model = self.train_model(target_name, target_column, features)
                results[target_name] = {
                    'model': model,
                    'metrics': self.evaluate_model(model, target_name)
                }
            except Exception as e:
                logger.error(f"Error training {target_name} model: {str(e)}")
                raise
                
        return results 

    def evaluate_model(self, model: Any, target_name: str) -> Dict[str, float]:
        """Evaluate model performance with multiple metrics.
        
        Args:
            model: Trained model
            target_name: Name of the target
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        X, y = self.prepare_features(self.features)
        predictions = model.predict(xgb.DMatrix(X))
        
        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        
        # Calculate prediction intervals
        std_dev = np.std(y - predictions)
        lower_bound = predictions - 1.96 * std_dev
        upper_bound = predictions + 1.96 * std_dev
        
        # Calculate coverage
        coverage = np.mean((y >= lower_bound) & (y <= upper_bound)) * 100
        
        # Calculate directional accuracy
        direction_correct = np.mean(np.sign(y.diff()) == np.sign(pd.Series(predictions).diff()))
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'coverage_95': coverage,
            'direction_accuracy': direction_correct
        }
        
        # Log metrics
        self.logger.info(f"\nModel evaluation metrics for {target_name}:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                f"{target_name}_metrics": metrics,
                f"{target_name}_predictions": wandb.Table(
                    data=[[true, pred, lb, ub] for true, pred, lb, ub in zip(y, predictions, lower_bound, upper_bound)],
                    columns=["true", "predicted", "lower_bound", "upper_bound"]
                )
            })
        
        return metrics
    
    def predict(self, features: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Make predictions with confidence intervals.
        
        Args:
            features: Dictionary of feature DataFrames
            
        Returns:
            Dictionary of predictions and confidence intervals
        """
        # Prepare features
        X, _ = self.prepare_features(features)
        
        # Get predictions
        predictions = self.model.predict(xgb.DMatrix(X))
        
        # Calculate prediction intervals
        std_dev = np.std(self.model.predict(xgb.DMatrix(self.X_train)) - self.y_train)
        lower_bound = predictions - 1.96 * std_dev
        upper_bound = predictions + 1.96 * std_dev
        
        # Calculate confidence scores
        confidence = 1 - (upper_bound - lower_bound) / (2 * std_dev)
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence': confidence
        } 