import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# Constants
DATA_BASE_DIR = "C:/Projects/NBA_Prediction"
PROCESSED_DIR = os.path.join(DATA_BASE_DIR, "data/processed")
FEATURES_DIR = os.path.join(PROCESSED_DIR, "features")

class DataValidator:
    def __init__(self):
        self.features = None
        self.validation_results = {}
        
    def load_features(self) -> None:
        """Load the processed features."""
        features_path = os.path.join(FEATURES_DIR, "game_features.csv")
        if os.path.exists(features_path):
            self.features = pd.read_csv(features_path)
        else:
            raise FileNotFoundError(f"Features file not found at {features_path}")
    
    def check_missing_values(self) -> Dict[str, float]:
        """Check for missing values in the dataset."""
        print("🔍 Checking for missing values...")
        
        missing = self.features.isnull().sum()
        missing_pct = (missing / len(self.features)) * 100
        
        # Get columns with missing values
        missing_cols = missing_pct[missing_pct > 0]
        
        if len(missing_cols) > 0:
            print("⚠️ Found missing values in the following columns:")
            for col, pct in missing_cols.items():
                print(f"  - {col}: {pct:.2f}% missing")
        
        return missing_cols.to_dict()
    
    def check_data_types(self) -> Dict[str, str]:
        """Check data types of columns."""
        print("🔍 Checking data types...")
        
        dtypes = self.features.dtypes
        unexpected_types = {}
        
        # Define expected types for key columns
        expected_types = {
            'game_id': 'object',
            'date': 'object',
            'home_team': 'object',
            'away_team': 'object',
            'home_team_id': 'int64',
            'away_team_id': 'int64'
        }
        
        for col, expected_type in expected_types.items():
            if col in dtypes and str(dtypes[col]) != expected_type:
                unexpected_types[col] = f"Expected {expected_type}, got {dtypes[col]}"
        
        if unexpected_types:
            print("⚠️ Found unexpected data types:")
            for col, msg in unexpected_types.items():
                print(f"  - {col}: {msg}")
        
        return unexpected_types
    
    def check_date_range(self) -> Dict[str, str]:
        """Check if dates are within expected range."""
        print("📅 Checking date range...")
        
        issues = {}
        
        # Convert to datetime
        self.features['date'] = pd.to_datetime(self.features['date'])
        
        # Check for future dates
        future_dates = self.features[self.features['date'] > pd.Timestamp.now()]
        if not future_dates.empty:
            issues['future_dates'] = f"Found {len(future_dates)} future dates"
        
        # Check for dates before season start
        season_start = pd.Timestamp('2024-10-04')
        old_dates = self.features[self.features['date'] < season_start]
        if not old_dates.empty:
            issues['old_dates'] = f"Found {len(old_dates)} dates before season start"
        
        if issues:
            print("⚠️ Found date range issues:")
            for issue, msg in issues.items():
                print(f"  - {issue}: {msg}")
        
        return issues
    
    def check_team_consistency(self) -> Dict[str, int]:
        """Check for team name and ID consistency."""
        print("🏀 Checking team consistency...")
        
        issues = {}
        
        # Check for duplicate team IDs
        team_id_counts = self.features['home_team_id'].value_counts()
        duplicate_teams = team_id_counts[team_id_counts > 1]
        if not duplicate_teams.empty:
            issues['duplicate_team_ids'] = len(duplicate_teams)
        
        # Check for missing team names
        missing_names = self.features['home_team'].isnull().sum()
        if missing_names > 0:
            issues['missing_team_names'] = missing_names
        
        if issues:
            print("⚠️ Found team consistency issues:")
            for issue, count in issues.items():
                print(f"  - {issue}: {count} instances")
        
        return issues
    
    def validate_data(self) -> Dict[str, Dict]:
        """Run all validation checks."""
        print("🚀 Starting data validation...")
        
        self.load_features()
        
        self.validation_results = {
            'missing_values': self.check_missing_values(),
            'data_types': self.check_data_types(),
            'date_range': self.check_date_range(),
            'team_consistency': self.check_team_consistency()
        }
        
        # Check if any issues were found
        has_issues = any(bool(results) for results in self.validation_results.values())
        
        if has_issues:
            print("\n⚠️ Validation completed with issues")
        else:
            print("\n✅ Validation completed successfully")
        
        return self.validation_results

if __name__ == "__main__":
    validator = DataValidator()
    validator.validate_data() 