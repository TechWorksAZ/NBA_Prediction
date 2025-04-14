import pytest
import pandas as pd
import numpy as np
from scripts.process.engineer_features import create_rolling_features, process_betting_data

def test_create_rolling_features():
    # Create sample data
    data = pd.DataFrame({
        'team': ['LAL', 'LAL', 'LAL', 'LAL', 'LAL'],
        'date': pd.date_range('2025-01-01', periods=5),
        'points': [100, 110, 105, 115, 120]
    })
    
    # Test rolling features
    result = create_rolling_features(data, ['points'], windows=[3])
    
    # Check if rolling features are created
    assert 'points_rolling_3_mean' in result.columns
    assert not result['points_rolling_3_mean'].isna().all()

def test_process_betting_data():
    # Create sample betting data
    data = pd.DataFrame({
        'game_id': ['001', '002', '003'],
        'date': pd.date_range('2025-01-01', periods=3),
        'away_team': ['LAL', 'GSW', 'BOS'],
        'home_team': ['BOS', 'LAL', 'GSW'],
        'away_spread': [-5.5, 2.5, -3.5],
        'home_spread': [5.5, -2.5, 3.5],
        'total': [220.5, 215.5, 225.5]
    })
    
    # Test betting data processing
    result = process_betting_data(data)
    
    # Check if processed columns exist
    assert 'closing_spread' in result.columns
    assert 'closing_total' in result.columns
    assert not result['closing_spread'].isna().all() 