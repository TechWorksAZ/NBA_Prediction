import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler

# Constants
DATA_BASE_DIR = "C:/Projects/NBA_Prediction"
RAW_DIR = os.path.join(DATA_BASE_DIR, "data/raw")
PROCESSED_DIR = os.path.join(DATA_BASE_DIR, "data/processed")
FEATURES_DIR = os.path.join(PROCESSED_DIR, "features")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

class FeatureEngineer:
    def __init__(self):
        self.core_data = None
        self.advanced_data = None
        self.defense_data = None
        self.matchup_data = None
        self.betting_data = None
        self.tracking_data = None
        self.team_features = None
        self.player_features = None
        self.combined_features = None
        
    def load_data(self) -> None:
        """Load all raw data categories."""
        print("📂 Loading raw data...")
        
        # Core data (basic stats)
        core_path = os.path.join(RAW_DIR, "core")
        if os.path.exists(core_path):
            self.core_data = {
                'team': pd.read_csv(os.path.join(core_path, "team_stats.csv")),
                'player': pd.read_csv(os.path.join(core_path, "player_stats.csv"))
            }
        
        # Advanced data
        advanced_path = os.path.join(RAW_DIR, "advanced")
        if os.path.exists(advanced_path):
            self.advanced_data = {
                'team': pd.read_csv(os.path.join(advanced_path, "team_advanced.csv")),
                'player': pd.read_csv(os.path.join(advanced_path, "player_advanced.csv"))
            }
        
        # Defense data
        defense_path = os.path.join(RAW_DIR, "defense")
        if os.path.exists(defense_path):
            self.defense_data = {
                'team': pd.read_csv(os.path.join(defense_path, "team_defense.csv")),
                'player': pd.read_csv(os.path.join(defense_path, "player_defense.csv"))
            }
        
        # Matchup and betting data
        betting_path = os.path.join(RAW_DIR, "betting")
        if os.path.exists(betting_path):
            self.betting_data = pd.read_csv(os.path.join(betting_path, "nba_sbr_odds_2025.csv"))
            self.matchup_data = pd.read_csv(os.path.join(betting_path, "historical_matchups.csv"))
        
        # Tracking data
        tracking_path = os.path.join(RAW_DIR, "tracking")
        if os.path.exists(tracking_path):
            self.tracking_data = {
                'team': pd.read_csv(os.path.join(tracking_path, "team_tracking.csv")),
                'player': pd.read_csv(os.path.join(tracking_path, "player_tracking.csv"))
            }
    
    def process_team_features(self) -> pd.DataFrame:
        """Process and combine all team-level features."""
        print("🏀 Processing team features...")
        
        # Initialize with core team stats
        team_features = self.core_data['team'].copy()
        
        # Add advanced stats
        if self.advanced_data is not None:
            team_features = pd.merge(
                team_features,
                self.advanced_data['team'],
                on=['team_id', 'date'],
                how='left'
            )
        
        # Add defense stats
        if self.defense_data is not None:
            team_features = pd.merge(
                team_features,
                self.defense_data['team'],
                on=['team_id', 'date'],
                how='left'
            )
        
        # Add tracking stats
        if self.tracking_data is not None:
            team_features = pd.merge(
                team_features,
                self.tracking_data['team'],
                on=['team_id', 'date'],
                how='left'
            )
        
        # Calculate rolling averages
        team_features = self._calculate_rolling_averages(team_features)
        
        # Calculate opponent-adjusted stats
        team_features = self._calculate_opponent_adjusted_stats(team_features)
        
        return team_features
    
    def process_player_features(self) -> pd.DataFrame:
        """Process and combine all player-level features."""
        print("👤 Processing player features...")
        
        # Initialize with core player stats
        player_features = self.core_data['player'].copy()
        
        # Add advanced stats
        if self.advanced_data is not None:
            player_features = pd.merge(
                player_features,
                self.advanced_data['player'],
                on=['player_id', 'date'],
                how='left'
            )
        
        # Add defense stats
        if self.defense_data is not None:
            player_features = pd.merge(
                player_features,
                self.defense_data['player'],
                on=['player_id', 'date'],
                how='left'
            )
        
        # Add tracking stats
        if self.tracking_data is not None:
            player_features = pd.merge(
                player_features,
                self.tracking_data['player'],
                on=['player_id', 'date'],
                how='left'
            )
        
        # Calculate rolling averages
        player_features = self._calculate_rolling_averages(player_features)
        
        # Calculate per-possession stats
        player_features = self._calculate_per_possession_stats(player_features)
        
        return player_features
    
    def process_betting_features(self) -> pd.DataFrame:
        """Process betting and matchup features."""
        print("💰 Processing betting features...")
        
        betting_features = self.betting_data.copy()
        
        # Calculate implied probabilities
        betting_features['home_implied_prob'] = betting_features['home_moneyline'].apply(
            lambda x: 100 / (x + 100) if x > 0 else abs(x) / (abs(x) + 100)
        )
        betting_features['away_implied_prob'] = betting_features['away_moneyline'].apply(
            lambda x: 100 / (x + 100) if x > 0 else abs(x) / (abs(x) + 100)
        )
        
        # Calculate line movement
        betting_features = self._calculate_line_movement(betting_features)
        
        # Add historical matchup data
        if self.matchup_data is not None:
            betting_features = pd.merge(
                betting_features,
                self.matchup_data,
                on=['home_team', 'away_team'],
                how='left'
            )
        
        return betting_features
    
    def _calculate_rolling_averages(self, df: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """Calculate rolling averages for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for window in windows:
            for col in numeric_cols:
                df[f'{col}_{window}g'] = df.groupby('team_id')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
        return df
    
    def _calculate_opponent_adjusted_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate opponent-adjusted statistics."""
        # This would require historical opponent data
        # For now, we'll just return the unadjusted stats
        return df
    
    def _calculate_per_possession_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-possession statistics."""
        # Add possession calculation
        df['possessions'] = df['fga'] - df['oreb'] + df['tov'] + 0.44 * df['fta']
        
        # Calculate per-possession stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['possessions', 'player_id', 'team_id']:
                df[f'{col}_per100'] = (df[col] / df['possessions']) * 100
        
        return df
    
    def _calculate_line_movement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate betting line movement features."""
        # Sort by game_id and time
        df = df.sort_values(['game_id', 'timestamp'])
        
        # Calculate line movement
        df['spread_movement'] = df.groupby('game_id')['spread'].diff()
        df['total_movement'] = df.groupby('game_id')['total'].diff()
        
        return df
    
    def combine_features(self) -> pd.DataFrame:
        """Combine all features into a single dataset."""
        print("🔄 Combining all features...")
        
        # Process each category
        team_features = self.process_team_features()
        player_features = self.process_player_features()
        betting_features = self.process_betting_features()
        
        # Combine team and betting features
        combined_features = pd.merge(
            team_features,
            betting_features,
            on=['team_id', 'date'],
            how='left'
        )
        
        # Add player features (aggregated by team)
        player_agg = player_features.groupby(['team_id', 'date']).agg({
            'points': 'mean',
            'assists': 'mean',
            'rebounds': 'mean',
            # Add more aggregations as needed
        }).reset_index()
        
        combined_features = pd.merge(
            combined_features,
            player_agg,
            on=['team_id', 'date'],
            how='left'
        )
        
        # Scale numeric features
        numeric_cols = combined_features.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        combined_features[numeric_cols] = scaler.fit_transform(combined_features[numeric_cols])
        
        # Save features
        output_path = os.path.join(FEATURES_DIR, "combined_features.csv")
        combined_features.to_csv(output_path, index=False)
        print(f"💾 Saved combined features to {output_path}")
        
        return combined_features
    
    def engineer_features(self) -> pd.DataFrame:
        """Main function to engineer all features."""
        print("🚀 Starting feature engineering...")
        
        # Load data
        self.load_data()
        
        # Combine and process features
        features = self.combine_features()
        
        print("✅ Feature engineering complete!")
        return features

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.engineer_features() 