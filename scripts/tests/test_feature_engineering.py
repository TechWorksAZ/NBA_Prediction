"""
Test suite for feature engineering functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scripts.feature_engineering import FeatureEngineer

@pytest.fixture
def sample_games_data():
    """Create sample games data for testing."""
    return pd.DataFrame({
        'game_id': ['20230101_ATL_BOS', '20230102_BOS_ATL'],
        'game_date': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        'home_team_id': [1, 2],
        'away_team_id': [2, 1],
        'home_team_abbr': ['ATL', 'BOS'],
        'away_team_abbr': ['BOS', 'ATL'],
        'home_score': [110, 105],
        'away_score': [105, 110]
    })

@pytest.fixture
def sample_advanced_team_data():
    """Create sample advanced team data for testing."""
    return pd.DataFrame({
        'game_id': ['20230101_ATL_BOS', '20230102_BOS_ATL'],
        'team_id': [1, 2],
        'off_rating': [115.5, 112.3],
        'def_rating': [108.2, 110.5],
        'net_rating': [7.3, 1.8],
        'pace': [100.5, 98.7]
    })

@pytest.fixture
def sample_tracking_team_data():
    """Create sample tracking team data for testing."""
    return pd.DataFrame({
        'game_id': ['20230101_ATL_BOS', '20230102_BOS_ATL'],
        'team_id': [1, 2],
        'avg_speed': [4.5, 4.3],
        'avg_distance': [2.1, 2.0],
        'avg_offensive_rating': [115.5, 112.3],
        'avg_defensive_rating': [108.2, 110.5]
    })

def test_calculate_team_ratings(sample_games_data):
    """Test team ratings calculation."""
    engineer = FeatureEngineer()
    ratings = engineer.calculate_team_ratings(sample_games_data)
    
    # Check if required columns are present
    assert 'offensive_rating' in ratings.columns
    assert 'defensive_rating' in ratings.columns
    assert 'net_rating' in ratings.columns
    
    # Check if ratings are calculated correctly
    assert not ratings['offensive_rating'].isna().any()
    assert not ratings['defensive_rating'].isna().any()
    assert not ratings['net_rating'].isna().any()

def test_calculate_rolling_stats(sample_games_data):
    """Test rolling statistics calculation."""
    engineer = FeatureEngineer()
    rolling_stats = engineer.calculate_rolling_stats(
        sample_games_data,
        ['team_id'],
        ['team_score', 'opponent_score'],
        [3, 5, 10]
    )
    
    # Check if rolling columns are created
    for window in [3, 5, 10]:
        assert f'team_score_rolling_{window}' in rolling_stats.columns
        assert f'opponent_score_rolling_{window}' in rolling_stats.columns
    
    # Check if values are calculated
    assert not rolling_stats[f'team_score_rolling_3'].isna().all()
    assert not rolling_stats[f'opponent_score_rolling_3'].isna().all()

def test_calculate_game_context_features(sample_games_data):
    """Test game context features calculation."""
    engineer = FeatureEngineer()
    context_features = engineer.calculate_game_context_features(sample_games_data)
    
    # Check if required columns are present
    assert 'back_to_back' in context_features.columns
    assert 'travel_distance' in context_features.columns
    assert 'rest_days' in context_features.columns
    assert 'rest_advantage' in context_features.columns
    
    # Check if values are calculated
    assert not context_features['back_to_back'].isna().all()
    assert not context_features['travel_distance'].isna().all()
    assert not context_features['rest_days'].isna().all()

def test_calculate_time_features(sample_games_data):
    """Test time-based features calculation."""
    engineer = FeatureEngineer()
    time_features = engineer.calculate_time_features(sample_games_data)
    
    # Check if required columns are present
    assert 'month' in time_features.columns
    assert 'day_of_week' in time_features.columns
    assert 'is_weekend' in time_features.columns
    assert 'season' in time_features.columns
    assert 'days_into_season' in time_features.columns
    assert 'season_progress' in time_features.columns
    
    # Check if values are calculated correctly
    assert time_features['month'].isin(range(1, 13)).all()
    assert time_features['day_of_week'].isin(range(7)).all()
    assert time_features['is_weekend'].isin([0, 1]).all()
    assert not time_features['season'].isna().any()
    assert not time_features['days_into_season'].isna().any()
    assert not time_features['season_progress'].isna().any()

def test_process_team_features(sample_games_data, sample_advanced_team_data, sample_tracking_team_data):
    """Test team features processing."""
    engineer = FeatureEngineer()
    engineer.games = sample_games_data
    engineer.advanced_team = sample_advanced_team_data
    engineer.tracking_team = sample_tracking_team_data
    
    team_features = engineer.process_team_features()
    
    # Check if required columns are present
    assert 'team_id' in team_features.columns
    assert 'opponent_team_id' in team_features.columns
    assert 'team_score' in team_features.columns
    assert 'opponent_score' in team_features.columns
    assert 'is_home' in team_features.columns
    
    # Check if advanced stats are merged
    assert 'off_rating' in team_features.columns
    assert 'def_rating' in team_features.columns
    assert 'net_rating' in team_features.columns
    assert 'pace' in team_features.columns
    
    # Check if tracking stats are merged
    assert 'avg_speed' in team_features.columns
    assert 'avg_distance' in team_features.columns
    assert 'avg_offensive_rating' in team_features.columns
    assert 'avg_defensive_rating' in team_features.columns

def test_process_game_features(sample_games_data, sample_advanced_team_data):
    """Test game features processing."""
    engineer = FeatureEngineer()
    engineer.games = sample_games_data
    engineer.advanced_team = sample_advanced_team_data
    
    game_features = engineer.process_game_features()
    
    # Check if required columns are present
    assert 'game_id' in game_features.columns
    assert 'game_date' in game_features.columns
    assert 'team_id' in game_features.columns
    assert 'opponent_team_id' in game_features.columns
    assert 'team_score' in game_features.columns
    assert 'opponent_score' in game_features.columns
    
    # Check if advanced features are calculated
    assert 'off_rating' in game_features.columns
    assert 'def_rating' in game_features.columns
    assert 'net_rating' in game_features.columns
    assert 'pace' in game_features.columns
    
    # Check if context features are calculated
    assert 'back_to_back' in game_features.columns
    assert 'travel_distance' in game_features.columns
    assert 'rest_days' in game_features.columns
    
    # Check if time features are calculated
    assert 'month' in game_features.columns
    assert 'day_of_week' in game_features.columns
    assert 'is_weekend' in game_features.columns
    assert 'season' in game_features.columns
    assert 'days_into_season' in game_features.columns
    assert 'season_progress' in game_features.columns

def test_analyze_feature_importance(sample_games_data, sample_advanced_team_data):
    """Test feature importance analysis."""
    engineer = FeatureEngineer()
    engineer.games = sample_games_data
    engineer.advanced_team = sample_advanced_team_data
    
    # Process game features
    game_features = engineer.process_game_features()
    
    # Add a target column for testing
    game_features['target'] = game_features['team_score'] - game_features['opponent_score']
    
    # Analyze feature importance
    importance_df = engineer.analyze_feature_importance(
        game_features,
        target_col='target',
        exclude_cols=['game_id', 'game_date', 'team_id', 'opponent_team_id']
    )
    
    # Check if importance DataFrame has required columns
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns
    assert 'cumulative_importance' in importance_df.columns
    assert 'importance_category' in importance_df.columns
    
    # Check if importance values are valid
    assert importance_df['importance'].between(0, 1).all()
    assert importance_df['cumulative_importance'].between(0, 1).all()
    assert importance_df['importance_category'].isin(['Low', 'Medium', 'High']).all()
    
    # Check if features are sorted by importance
    assert importance_df['importance'].is_monotonic_decreasing

def test_validate_features(sample_games_data, sample_advanced_team_data):
    """Test feature validation."""
    engineer = FeatureEngineer()
    engineer.games = sample_games_data
    engineer.advanced_team = sample_advanced_team_data
    
    # Process game features
    game_features = engineer.process_game_features()
    
    # Add some test cases
    game_features['missing_values'] = [np.nan, 1]  # Add missing values
    game_features['infinite_values'] = [np.inf, 1]  # Add infinite values
    game_features['outliers'] = [1000, 1]  # Add outliers
    game_features['correlated'] = game_features['team_score'] * 1.1  # Add correlated feature
    game_features['constant'] = 1  # Add constant feature
    
    # Validate features
    validation_results = engineer.validate_features(game_features)
    
    # Check validation results
    assert 'missing_values' in validation_results
    assert 'infinite_values' in validation_results
    assert 'outliers' in validation_results
    assert 'correlated_features' in validation_results
    assert 'constant_features' in validation_results
    
    # Check specific issues
    assert 'missing_values' in validation_results['missing_values']
    assert 'infinite_values' in validation_results['infinite_values']
    assert 'outliers' in validation_results['outliers']
    assert 'correlated' in validation_results['correlated_features']
    assert 'constant' in validation_results['constant_features']

def test_process_player_features(sample_player_features):
    """Test the process_player_features method."""
    engineer = FeatureEngineer()
    engineer.player_features = sample_player_features
    
    result = engineer.process_player_features()
    
    # Check that the result is not empty
    assert not result.empty
    
    # Check required columns are present
    required_columns = [
        'GAME_ID', 'TEAM_ID', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
        'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'PLAYER_EFF', 'USG_PCT',
        'TEAM_FG_PCT', 'TEAM_3P_PCT', 'TEAM_FT_PCT', 'TEAM_TS_PCT',
        'TEAM_EFG_PCT', 'TEAM_ORtg'
    ]
    assert all(col in result.columns for col in required_columns)
    
    # Check that percentages are between 0 and 1
    percentage_columns = [
        'TEAM_FG_PCT', 'TEAM_3P_PCT', 'TEAM_FT_PCT', 'TEAM_TS_PCT',
        'TEAM_EFG_PCT', 'USG_PCT'
    ]
    for col in percentage_columns:
        assert result[col].between(0, 1).all()
    
    # Check that counting stats are non-negative
    counting_columns = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA']
    for col in counting_columns:
        assert (result[col] >= 0).all()
    
    # Check that minutes are reasonable (0-240)
    assert result['MIN'].between(0, 240).all()
    
    # Check that offensive rating is reasonable
    assert result['TEAM_ORtg'].between(0, 200).all() 