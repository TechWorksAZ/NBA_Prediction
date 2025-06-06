"""
Total Points Model Trainer

This module implements the total points prediction model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, List
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb
from .evaluation import ModelEvaluator

logger = logging.getLogger(__name__)

class TotalModelTrainer:
    """Trainer for total points prediction model."""
    
    def __init__(self, use_wandb: bool = True):
        """Initialize the total model trainer.
        
        Args:
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.use_wandb = use_wandb
        self.model = None
        self.scaler = StandardScaler()
        self.evaluator = ModelEvaluator('total_model', use_wandb=use_wandb)
        
    def _create_rolling_features(self, df: pd.DataFrame, window_sizes: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """Create rolling average features for numeric columns.
        
        Args:
            df: DataFrame to create features from
            window_sizes: List of window sizes for rolling averages
            
        Returns:
            DataFrame with additional rolling features
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        rolling_features = df.copy()
        
        for window in window_sizes:
            for col in numeric_cols:
                rolling_features[f'{col}_rolling_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                
        return rolling_features
        
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key statistics.
        
        Args:
            df: DataFrame to create features from
            
        Returns:
            DataFrame with additional interaction features
        """
        interactions = df.copy()
        
        # Offense-defense interactions
        if all(col in df.columns for col in ['OFF_RATING', 'DEF_RATING']):
            interactions['OFF_DEF_RATIO'] = df['OFF_RATING'] / df['DEF_RATING']
            
        # Pace and efficiency interactions
        if all(col in df.columns for col in ['PACE', 'EFG_PCT']):
            interactions['PACE_EFF'] = df['PACE'] * df['EFG_PCT']
            
        # Turnover and shooting interactions
        if all(col in df.columns for col in ['TOV_PCT', 'TS_PCT']):
            interactions['TOV_EFF'] = df['TOV_PCT'] * df['TS_PCT']
            
        return interactions
    
    def prepare_features(self, features: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for training.
        
        Args:
            features: Dictionary of feature DataFrames
            
        Returns:
            Tuple of (X, y) for training
        """
        # Get required dataframes
        team_game = features['team_game'].copy()
        target = features['target'].copy()
        betting = features['betting'].copy()
        
        # Convert GAME_ID to string format in all dataframes
        for df in [team_game, target, betting]:
            df['GAME_ID'] = df['GAME_ID'].astype(str).str.replace('.0', '')
            
        # Sort by date if available
        if 'GAME_DATE' in team_game.columns:
            team_game = team_game.sort_values('GAME_DATE')
        
        # Create rolling features
        team_game = self._create_rolling_features(team_game)
        
        # Create interaction features
        team_game = self._create_interaction_features(team_game)
        
        # Log shapes before merge
        logger.info(f"Team game data shape: {team_game.shape}")
        logger.info(f"Target data shape: {target.shape}")
        logger.info(f"Betting data shape: {betting.shape}")
        
        # Log sample GAME_IDs for debugging
        logger.info(f"Team game GAME_ID sample: {team_game['GAME_ID'].head().tolist()}")
        logger.info(f"Target GAME_ID sample: {target['GAME_ID'].head().tolist()}")
        logger.info(f"Betting GAME_ID sample: {betting['GAME_ID'].head().tolist()}")
        
        # Merge data
        merged = pd.merge(team_game, target, on='GAME_ID', how='inner')
        merged = pd.merge(merged, betting, on='GAME_ID', how='inner')
        
        # Log shape after merge
        logger.info(f"Shape after merge: {merged.shape}")
        
        if merged.empty:
            raise ValueError("Merged DataFrame is empty after merging on GAME_ID! Check logs for diagnostics.")
            
        # Prepare features and target
        feature_cols = [col for col in merged.columns if col not in [
            'GAME_ID', 'TARGET_TOTAL', 'GAME_DATE', 'HOME_SCORE', 'AWAY_SCORE',
            'TARGET_WIN', 'TARGET_SPREAD', 'TARGET_OVER'
        ]]
        X = merged[feature_cols]
        y = merged['TARGET_TOTAL']

        # Drop object (string) columns from X
        object_cols = X.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            logger.info(f"Dropping non-numeric columns from features: {object_cols}")
            X = X.drop(columns=object_cols)

        # Handle missing values
        X = X.fillna(X.mean())

        # Scale features
        X = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Log feature information
        logger.info(f"Number of features: {X.shape[1]}")
        logger.info(f"Feature names: {X.columns.tolist()}")

        if self.use_wandb:
            wandb.log({
                "n_features": X.shape[1],
                "feature_names": X.columns.tolist()
            })
        
        return X, y
    
    def train(self, features: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train the total points model.
        
        Args:
            features: Dictionary of feature DataFrames
            
        Returns:
            Dictionary of training results
        """
        # Prepare features
        X, y = self.prepare_features(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Log split sizes
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Initialize model with optimized parameters
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            eval_metric=['mae', 'rmse'],
            early_stopping_rounds=20
        )
        
        # Train model
        logger.info("Training total points model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=True
        )
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluate model
        metrics = self.evaluator.evaluate_regression(y_test, y_pred)
        
        # Plot predictions
        self.evaluator.plot_predictions(y_test, y_pred)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 most important features:")
        logger.info(importance.head(10))
        
        if self.use_wandb:
            wandb.log({
                "feature_importance": wandb.Table(
                    dataframe=importance.head(10)
                ),
                "best_iteration": self.model.best_iteration,
                "best_score": self.model.best_score
            })
            
        # Save metrics
        self.evaluator.save_metrics(Path('C:/Projects/NBA_Prediction/models/metrics'))
        
        return {
            'metrics': metrics,
            'feature_importance': importance.to_dict('records'),
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score
        }
        
    def predict(self, features: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            features: Dictionary of feature DataFrames
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
            
        X, _ = self.prepare_features(features)
        return self.model.predict(X) 