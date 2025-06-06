"""
NBA Model Training Pipeline

This script orchestrates the training of multiple NBA prediction models:
1. Game outcomes and spreads
2. Over/Under totals
3. Half-time/Quarter predictions
4. Player prop predictions

The pipeline:
1. Loads processed features from feature pipeline
2. Trains each model type with appropriate features and targets
3. Evaluates model performance
4. Saves trained models and metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any
import json
from datetime import datetime
import wandb
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import model trainers
from models.model_trainer import NBAModelTraining
from models.game_model import GameModelTrainer
from models.total_model import TotalModelTrainer
from models.period_model import PeriodModelTrainer
from models.player_prop_model import PlayerPropModelTrainer
from models.test_model_trainer import TestModelTrainer

# Create logs directory if it doesn't exist
base_dir = Path('C:/Projects/NBA_Prediction')
logs_dir = base_dir / 'logs'
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'model_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelPipeline:
    """Pipeline for training NBA prediction models."""
    
    def __init__(self, use_wandb: bool = True):
        """Initialize the model pipeline.
        
        Args:
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.base_dir = Path('C:/Projects/NBA_Prediction')
        self.data_dir = self.base_dir / 'data'
        self.features_dir = self.data_dir / 'processed' / 'features'
        self.models_dir = self.base_dir / 'models'
        self.use_wandb = use_wandb
        
        # Initialize model trainers
        self.trainers = {
            'game': GameModelTrainer(use_wandb=use_wandb),
            'total': TotalModelTrainer(use_wandb=use_wandb),
            'period': PeriodModelTrainer(use_wandb=use_wandb),
            'player_prop': PlayerPropModelTrainer(use_wandb=use_wandb),
            'test': TestModelTrainer(use_wandb=use_wandb)
        }
        
        # Initialize wandb if enabled
        if use_wandb:
                wandb.init(
                    project="nba-prediction",
                    config={
                    "base_dir": str(self.base_dir),
                    "data_dir": str(self.data_dir),
                    "features_dir": str(self.features_dir),
                    "models_dir": str(self.models_dir)
        }
            )
    
    def load_features(self) -> Dict[str, pd.DataFrame]:
        """Load feature files from the processed data directory.
        
        Returns:
            Dictionary mapping feature names to DataFrames
        """
        features = {}
        feature_files = {
            'team_game': 'team_game.csv',
            'player_game': 'player_game.csv',
            'team_specific': 'team_specific.csv',
            'advanced': 'advanced.csv',
            'shotchart': 'shotchart.csv',
            'betting': 'betting.csv',
            'target': 'target.csv'
        }
        
        for name, file in feature_files.items():
                features[name] = pd.read_csv(self.features_dir / file)
            logging.info(f"Loaded {file}: {len(features[name])} rows")
                
            # Log feature info to wandb
                if self.use_wandb:
                    wandb.log({
                        f"{name}_rows": len(features[name]),
                        f"{name}_columns": len(features[name].columns)
                    })
        
        return features
    
    def train_models(self, features: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Train all models using the provided features.
        
        Args:
            features: Dictionary of feature DataFrames
            
        Returns:
            Dictionary mapping model names to their training results
        """
        results = {}
        
        for name, trainer in self.trainers.items():
            try:
                logging.info(f"\nTraining {name} model...")
                results[name] = trainer.train(features)
            except Exception as e:
                logging.error(f"Error training {name} model: {str(e)}")
                raise
        
        return results
    
    def run(self):
        """Run the complete model training pipeline."""
        try:
            logging.info("Starting model training pipeline...")
            
            # Load features
            logging.info("Loading feature files...")
            features = self.load_features()
            
            # Train models
            logging.info("Starting model training...")
            results = self.train_models(features)
            
            # Log results
            if self.use_wandb:
                for name, result in results.items():
                    wandb.log({f"{name}_results": result})
                    
            logging.info("Model training pipeline completed successfully!")
            
        except Exception as e:
            logging.error(f"Error in model training pipeline: {str(e)}")
            raise
        finally:
            if self.use_wandb:
                wandb.finish()

if __name__ == "__main__":
    pipeline = ModelPipeline(use_wandb=True)  # Enable wandb
    pipeline.run() 