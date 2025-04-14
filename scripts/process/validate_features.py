import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

# Constants
DATA_BASE_DIR = "C:/Projects/NBA_Prediction"
PROCESSED_DIR = os.path.join(DATA_BASE_DIR, "data/processed")
FEATURES_DIR = os.path.join(PROCESSED_DIR, "features")

class FeatureValidator:
    def __init__(self):
        self.features = None
        self.validation_results = {}
        
    def load_features(self) -> None:
        """Load the engineered features."""
        features_path = os.path.join(FEATURES_DIR, "combined_features.csv")
        if os.path.exists(features_path):
            self.features = pd.read_csv(features_path)
        else:
            raise FileNotFoundError(f"Features file not found at {features_path}")
    
    def check_feature_correlation(self, threshold: float = 0.95) -> Dict[str, List[str]]:
        """Check for highly correlated features."""
        print("🔍 Checking feature correlations...")
        
        # Get numeric columns
        numeric_cols = self.features.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        corr_matrix = self.features[numeric_cols].corr()
        
        # Find highly correlated features
        high_corr = {}
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    if col1 not in high_corr:
                        high_corr[col1] = []
                    high_corr[col1].append(col2)
        
        if high_corr:
            print("⚠️ Found highly correlated features:")
            for col, corr_cols in high_corr.items():
                print(f"  - {col} is highly correlated with: {', '.join(corr_cols)}")
        
        return high_corr
    
    def check_feature_importance(self, target_col: str = 'home_implied_prob') -> Dict[str, float]:
        """Check feature importance using mutual information."""
        print("📊 Checking feature importance...")
        
        # Get numeric columns
        numeric_cols = self.features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(
            self.features[numeric_cols],
            self.features[target_col]
        )
        
        # Create importance dictionary
        importance = dict(zip(numeric_cols, mi_scores))
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        # Print top 10 features
        print("Top 10 most important features:")
        for feature, score in list(importance.items())[:10]:
            print(f"  - {feature}: {score:.4f}")
        
        return importance
    
    def check_feature_redundancy(self, n_components: int = 10) -> Dict[str, float]:
        """Check for redundant features using PCA."""
        print("🔄 Checking feature redundancy...")
        
        # Get numeric columns
        numeric_cols = self.features.select_dtypes(include=[np.number]).columns
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca.fit(self.features[numeric_cols])
        
        # Calculate explained variance ratio
        explained_variance = dict(zip(
            numeric_cols,
            np.sum(pca.components_ ** 2, axis=0)
        ))
        
        # Sort by explained variance
        explained_variance = dict(sorted(
            explained_variance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        # Print top 10 features
        print("Top 10 features by explained variance:")
        for feature, var in list(explained_variance.items())[:10]:
            print(f"  - {feature}: {var:.4f}")
        
        return explained_variance
    
    def check_missing_values(self) -> Dict[str, float]:
        """Check for missing values in features."""
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
    
    def validate_features(self) -> Dict[str, Dict]:
        """Run all validation checks."""
        print("🚀 Starting feature validation...")
        
        self.load_features()
        
        self.validation_results = {
            'correlation': self.check_feature_correlation(),
            'importance': self.check_feature_importance(),
            'redundancy': self.check_feature_redundancy(),
            'missing_values': self.check_missing_values()
        }
        
        # Check if any issues were found
        has_issues = any(bool(results) for results in self.validation_results.values())
        
        if has_issues:
            print("\n⚠️ Validation completed with issues")
        else:
            print("\n✅ Validation completed successfully")
        
        return self.validation_results

if __name__ == "__main__":
    validator = FeatureValidator()
    validator.validate_features() 