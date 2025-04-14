import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

# Constants
DATA_BASE_DIR = "C:/Projects/NBA_Prediction"
RAW_DIR = os.path.join(DATA_BASE_DIR, "data/raw")
PROCESSED_DIR = os.path.join(DATA_BASE_DIR, "data/processed")
FEATURES_DIR = os.path.join(PROCESSED_DIR, "features")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

class DataProcessor:
    def __init__(self):
        self.team_stats = None
        self.games = None
        self.odds = None
        self.features = None
        
    def load_data(self, date: Optional[str] = None) -> None:
        """Load all necessary data for processing."""
        print("📂 Loading data...")
        
        # Load games data
        games_path = os.path.join(RAW_DIR, "core/nba_games.csv")
        if os.path.exists(games_path):
            self.games = pd.read_csv(games_path)
            if date:
                self.games = self.games[self.games['date'] == date]
        
        # Load odds data
        odds_path = os.path.join(RAW_DIR, "betting/nba_sbr_odds_2025.csv")
        if os.path.exists(odds_path):
            self.odds = pd.read_csv(odds_path)
            if date:
                self.odds = self.odds[self.odds['date'] == date]
        
        # Load team stats
        team_stats_path = os.path.join(RAW_DIR, "core/team_stats.csv")
        if os.path.exists(team_stats_path):
            self.team_stats = pd.read_csv(team_stats_path)
    
    def process_team_stats(self) -> pd.DataFrame:
        """Process team statistics into features."""
        print("📊 Processing team statistics...")
        
        if self.team_stats is None:
            raise ValueError("Team stats not loaded")
            
        # Calculate rolling averages
        stats_df = self.team_stats.copy()
        
        # Sort by team and date
        stats_df = stats_df.sort_values(['team_id', 'date'])
        
        # Calculate rolling averages (3, 5, 10 games)
        rolling_windows = [3, 5, 10]
        for window in rolling_windows:
            stats_df[f'off_rating_{window}g'] = stats_df.groupby('team_id')['off_rating'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            stats_df[f'def_rating_{window}g'] = stats_df.groupby('team_id')['def_rating'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            stats_df[f'net_rating_{window}g'] = stats_df.groupby('team_id')['net_rating'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        return stats_df
    
    def process_odds(self) -> pd.DataFrame:
        """Process betting odds into features."""
        print("💰 Processing betting odds...")
        
        if self.odds is None:
            raise ValueError("Odds data not loaded")
            
        odds_df = self.odds.copy()
        
        # Calculate implied probabilities from moneyline odds
        def implied_probability(odds):
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        
        # Process moneyline odds
        odds_df['home_implied_prob'] = odds_df['home_moneyline'].apply(implied_probability)
        odds_df['away_implied_prob'] = odds_df['away_moneyline'].apply(implied_probability)
        
        # Calculate closing line value (CLV)
        # This would need historical odds data to be meaningful
        # For now, we'll just store the current odds
        
        return odds_df
    
    def create_game_features(self) -> pd.DataFrame:
        """Create features for each game."""
        print("🎯 Creating game features...")
        
        if self.games is None:
            raise ValueError("Games data not loaded")
            
        games_df = self.games.copy()
        
        # Add rest days
        games_df['date'] = pd.to_datetime(games_df['date'])
        games_df = games_df.sort_values(['home_team_id', 'date'])
        
        # Calculate days since last game for home team
        games_df['home_rest_days'] = games_df.groupby('home_team_id')['date'].diff().dt.days
        games_df['home_rest_days'] = games_df['home_rest_days'].fillna(7)  # First game of season
        
        # Calculate days since last game for away team
        games_df = games_df.sort_values(['away_team_id', 'date'])
        games_df['away_rest_days'] = games_df.groupby('away_team_id')['date'].diff().dt.days
        games_df['away_rest_days'] = games_df['away_rest_days'].fillna(7)  # First game of season
        
        # Reset index
        games_df = games_df.sort_values(['date', 'game_id'])
        
        return games_df
    
    def merge_features(self) -> pd.DataFrame:
        """Merge all features into a single dataframe."""
        print("🔄 Merging all features...")
        
        # Get processed data
        team_stats = self.process_team_stats()
        odds = self.process_odds()
        games = self.create_game_features()
        
        # Merge team stats with games
        features = pd.merge(
            games,
            team_stats,
            left_on=['home_team_id', 'date'],
            right_on=['team_id', 'date'],
            suffixes=('', '_home')
        )
        
        features = pd.merge(
            features,
            team_stats,
            left_on=['away_team_id', 'date'],
            right_on=['team_id', 'date'],
            suffixes=('', '_away')
        )
        
        # Merge odds data
        features = pd.merge(
            features,
            odds,
            on=['game_id', 'date'],
            how='left'
        )
        
        # Save features
        output_path = os.path.join(FEATURES_DIR, "game_features.csv")
        features.to_csv(output_path, index=False)
        print(f"💾 Saved features to {output_path}")
        
        return features
    
    def process_data(self, date: Optional[str] = None) -> pd.DataFrame:
        """Main function to process all data."""
        print("🚀 Starting data processing...")
        
        # Load data
        self.load_data(date)
        
        # Process and merge features
        features = self.merge_features()
        
        print("✅ Data processing complete!")
        return features

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_data() 