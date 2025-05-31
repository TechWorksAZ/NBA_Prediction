"""
Advanced Feature Engineering for NBA Prediction Model

This script creates advanced features from the processed data to improve model accuracy.
It focuses on derived metrics, advanced statistics, and composite features that combine
multiple data sources.

Author: NBA Prediction Team
Date: 2024
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)

def create_advanced_features(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create advanced features from all available data sources."""
    logger.info("Creating advanced features...")
    
    # Initialize features DataFrame with game IDs
    features = pd.DataFrame()
    features['GAME_ID'] = data['pg_data']['GAME_ID'].unique()
    
    # 1. Shot Quality Metrics
    shot_quality = create_shot_quality_features(data)
    features = features.merge(shot_quality, on='GAME_ID', how='left')
    
    # 2. Lineup Impact Metrics
    lineup_impact = create_lineup_impact_features(data)
    features = features.merge(lineup_impact, on='GAME_ID', how='left')
    
    # 3. Game Flow Metrics
    game_flow = create_game_flow_features(data)
    features = features.merge(game_flow, on='GAME_ID', how='left')
    
    # 4. Matchup Advantage Metrics
    matchup_advantage = create_matchup_advantage_features(data)
    features = features.merge(matchup_advantage, on='GAME_ID', how='left')
    
    # 5. Advanced Team Metrics
    team_metrics = create_advanced_team_metrics(data)
    features = features.merge(team_metrics, on='GAME_ID', how='left')
    
    # Fill missing values with appropriate defaults
    numeric_columns = features.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != 'GAME_ID':
            if col in ['paint_shots_pct', 'midrange_shots_pct', 'corner_3_pct']:
                # For percentage columns, fill with league average
                features[col] = features[col].fillna(features[col].mean())
            elif col in ['lead_changes', 'max_lead', 'min_lead', 'lead_volatility']:
                # For game flow metrics, fill with 0
                features[col] = features[col].fillna(0)
            elif col in ['avg_matchup_advantage', 'matchup_efficiency']:
                # For matchup metrics, fill with league average
                features[col] = features[col].fillna(features[col].mean())
            elif col == 'starter_bench_plusminus_diff':
                # For plus/minus difference, fill with 0
                features[col] = features[col].fillna(0)
            else:
                # For other numeric columns, fill with mean
                features[col] = features[col].fillna(features[col].mean())
    
    logger.info("Advanced features created successfully")
    return features

def create_shot_quality_features(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create shot quality metrics from shot chart data."""
    logger.info("Starting shot quality feature creation...")
    
    # Initialize features with all game IDs from pg_data
    features = pd.DataFrame()
    features['GAME_ID'] = data['pg_data']['GAME_ID'].unique()
    logger.info(f"Total games to process: {len(features)}")
    
    if 'pg_shotchart_detail' in data:
        shots = data['pg_shotchart_detail'].copy()
        logger.info(f"Shot chart data shape: {shots.shape}")
        logger.info(f"Shot chart columns: {shots.columns.tolist()}")
        
        # Calculate shot quality metrics
        if 'SHOT_DISTANCE' in shots.columns:
            # Calculate average shot distance per game
            game_shots = shots.groupby('GAME_ID')['SHOT_DISTANCE'].agg(['mean', 'count'])
            features = features.merge(
                game_shots['mean'].reset_index().rename(columns={'mean': 'avg_shot_distance'}),
                on='GAME_ID',
                how='left'
            )
            
            # Fill missing values with league average
            league_avg_distance = features['avg_shot_distance'].mean()
            features['avg_shot_distance'] = features['avg_shot_distance'].fillna(league_avg_distance)
            
        if 'SHOT_MADE_FLAG' in shots.columns and 'SHOT_DISTANCE' in shots.columns:
            # Calculate shot quality score
            shots['shot_quality'] = shots['SHOT_MADE_FLAG'] * (1 + 0.5 * (shots['SHOT_DISTANCE'] > 23.75))
            game_quality = shots.groupby('GAME_ID')['shot_quality'].agg(['mean', 'count'])
            features = features.merge(
                game_quality['mean'].reset_index().rename(columns={'mean': 'shot_quality_score'}),
                on='GAME_ID',
                how='left'
            )
            
            # Fill missing values with league average
            league_avg_quality = features['shot_quality_score'].mean()
            features['shot_quality_score'] = features['shot_quality_score'].fillna(league_avg_quality)
            
        if 'SHOT_ZONE_BASIC' in shots.columns:
            # Calculate zone percentages
            total_shots = shots.groupby('GAME_ID').size()
            paint_shots = shots[shots['SHOT_ZONE_BASIC'] == 'In The Paint (Non-RA)'].groupby('GAME_ID').size()
            midrange_shots = shots[shots['SHOT_ZONE_BASIC'] == 'Mid-Range'].groupby('GAME_ID').size()
            corner_3_shots = shots[shots['SHOT_ZONE_BASIC'] == 'Corner 3'].groupby('GAME_ID').size()
            
            # Calculate league averages for each zone
            league_paint_pct = (paint_shots.sum() / total_shots.sum())
            league_midrange_pct = (midrange_shots.sum() / total_shots.sum())
            league_corner3_pct = (corner_3_shots.sum() / total_shots.sum())
            
            # Create zone percentage features with proper handling of missing values
            zone_features = pd.DataFrame(index=features['GAME_ID'])
            
            # Handle each zone separately to avoid division by zero
            for zone, shots_series, league_avg in [
                ('paint_shots_pct', paint_shots, league_paint_pct),
                ('midrange_shots_pct', midrange_shots, league_midrange_pct),
                ('corner_3_pct', corner_3_shots, league_corner3_pct)
            ]:
                # Calculate percentage only where total shots > 0
                zone_pct = pd.Series(index=total_shots.index, dtype=float)
                valid_games = total_shots > 0
                zone_pct[valid_games] = shots_series[valid_games] / total_shots[valid_games]
                zone_pct[~valid_games] = league_avg
                zone_features[zone] = zone_pct
            
            # Merge zone features
            features = features.merge(zone_features, on='GAME_ID', how='left')
    
    return features

def create_lineup_impact_features(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create lineup impact metrics from starter/bench data."""
    features = pd.DataFrame()
    features['GAME_ID'] = data['pg_data']['GAME_ID'].unique()
    
    if 'tg_starters' in data and 'tg_bench' in data:
        starters = data['tg_starters'].copy()
        bench = data['tg_bench'].copy()
        
        # Ensure numeric columns and handle any non-numeric values
        for df in [starters, bench]:
            for col in ['POINTS', 'MINUTES', 'FIELD_GOALS_ATTEMPTED', 'FREE_THROWS_ATTEMPTED', 
                       'ASSISTS', 'TURNOVERS', 'REBOUNDS_OFFENSIVE', 'REBOUNDS_DEFENSIVE']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate advanced metrics for both starters and bench
        for df in [starters, bench]:
            # Efficiency metrics
            df['efficiency'] = df['POINTS'] / (df['FIELD_GOALS_ATTEMPTED'] + 0.44 * df['FREE_THROWS_ATTEMPTED'])
            df['true_shooting'] = df['POINTS'] / (2 * (df['FIELD_GOALS_ATTEMPTED'] + 0.44 * df['FREE_THROWS_ATTEMPTED']))
            
            # Advanced stats
            df['assist_ratio'] = df['ASSISTS'] / (df['FIELD_GOALS_ATTEMPTED'] + 0.44 * df['FREE_THROWS_ATTEMPTED'] + df['TURNOVERS'])
            df['turnover_ratio'] = df['TURNOVERS'] / (df['FIELD_GOALS_ATTEMPTED'] + 0.44 * df['FREE_THROWS_ATTEMPTED'] + df['TURNOVERS'])
            df['usage_rate'] = (df['FIELD_GOALS_ATTEMPTED'] + 0.44 * df['FREE_THROWS_ATTEMPTED'] + df['TURNOVERS']) / df['MINUTES']
            
            # Rebounding metrics
            df['offensive_rebound_rate'] = df['REBOUNDS_OFFENSIVE'] / (df['REBOUNDS_OFFENSIVE'] + df['REBOUNDS_DEFENSIVE'])
        
        # Group by game and team to get totals
        starters_grouped = starters.groupby(['GAME_ID', 'TEAM_ID']).agg({
            'POINTS': 'sum',
            'MINUTES': 'sum',
            'efficiency': 'mean',
            'true_shooting': 'mean',
            'assist_ratio': 'mean',
            'turnover_ratio': 'mean',
            'usage_rate': 'mean',
            'offensive_rebound_rate': 'mean'
        }).reset_index()
        
        bench_grouped = bench.groupby(['GAME_ID', 'TEAM_ID']).agg({
            'POINTS': 'sum',
            'MINUTES': 'sum',
            'efficiency': 'mean',
            'true_shooting': 'mean',
            'assist_ratio': 'mean',
            'turnover_ratio': 'mean',
            'usage_rate': 'mean',
            'offensive_rebound_rate': 'mean'
        }).reset_index()
        
        # Calculate lineup impact metrics
        lineup_features = pd.DataFrame(index=features['GAME_ID'])
        
        # Points difference (keeping this as it's valuable)
        if 'POINTS' in starters_grouped.columns and 'POINTS' in bench_grouped.columns:
            lineup_data = pd.merge(
                starters_grouped[['GAME_ID', 'TEAM_ID', 'POINTS']],
                bench_grouped[['GAME_ID', 'TEAM_ID', 'POINTS']],
                on=['GAME_ID', 'TEAM_ID'],
                suffixes=('_starter', '_bench'),
                how='outer'
            )
            lineup_data['starter_bench_pts_diff'] = (
                lineup_data['POINTS_starter'].fillna(0) - lineup_data['POINTS_bench'].fillna(0)
            )
            lineup_features['starter_bench_pts_diff'] = lineup_data.groupby('GAME_ID')['starter_bench_pts_diff'].mean()
        
        # Efficiency differences between starters and bench
        for metric in ['efficiency', 'true_shooting', 'assist_ratio', 'turnover_ratio', 'usage_rate', 'offensive_rebound_rate']:
            # Calculate difference between starter and bench performance
            starters_grouped[f'{metric}_diff'] = starters_grouped[metric] - bench_grouped[metric]
            
            # Calculate the difference between teams for each metric
            game_metric_diff = starters_grouped.groupby('GAME_ID')[f'{metric}_diff'].diff()
            lineup_features[f'team_{metric}_diff'] = game_metric_diff
        
        # Minutes and scoring distribution
        lineup_features['starter_minutes_ratio'] = starters_grouped['MINUTES'] / (starters_grouped['MINUTES'] + bench_grouped['MINUTES'])
        lineup_features['bench_scoring_ratio'] = bench_grouped['POINTS'] / (starters_grouped['POINTS'] + bench_grouped['POINTS'])
        
        # Bench contribution ratio (keeping this as it's valuable)
        if 'POINTS' in lineup_data.columns:
            lineup_data['bench_contribution_ratio'] = np.where(
                lineup_data['POINTS_starter'] > 0,
                lineup_data['POINTS_bench'].fillna(0) / lineup_data['POINTS_starter'],
                0
            )
            lineup_features['bench_contribution_ratio'] = lineup_data.groupby('GAME_ID')['bench_contribution_ratio'].mean()
        
        # Merge lineup features
        features = features.merge(lineup_features, on='GAME_ID', how='left')
        
        # Fill missing values with appropriate defaults
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'GAME_ID':
                if col in ['starter_minutes_ratio', 'bench_scoring_ratio', 'bench_contribution_ratio']:
                    # For ratio columns, fill with 0.5 (50%)
                    features[col] = features[col].fillna(0.5)
                elif col.endswith('_diff'):
                    # For difference columns, fill with 0
                    features[col] = features[col].fillna(0)
                else:
                    # For other numeric columns, fill with mean
                    features[col] = features[col].fillna(features[col].mean())
    
    return features

def create_game_flow_features(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create game flow metrics from play-by-play data."""
    features = pd.DataFrame()
    features['GAME_ID'] = data['pg_data']['GAME_ID'].unique()
    
    if 'pg_pbp' in data:
        pbp = data['pg_pbp'].copy()
        pbp['SCORE_MARGIN'] = pd.to_numeric(pbp['SCORE_MARGIN'], errors='coerce')
        
        # Group by game
        game_pbp = pbp.groupby('GAME_ID')
        
        # Game flow metrics
        flow_features = pd.DataFrame(index=features['GAME_ID'])
        flow_features['lead_changes'] = game_pbp.apply(
            lambda x: (x['TEAM_LEADING'].shift() != x['TEAM_LEADING']).sum() if len(x) > 1 else 0
        )
        flow_features['max_lead'] = game_pbp['SCORE_MARGIN'].max()
        flow_features['min_lead'] = game_pbp['SCORE_MARGIN'].min()
        flow_features['lead_volatility'] = game_pbp['SCORE_MARGIN'].std()
        
        # Merge flow features
        features = features.merge(flow_features, on='GAME_ID', how='left')
    
    return features

def create_matchup_advantage_features(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create matchup advantage metrics from player matchup data."""
    features = pd.DataFrame()
    features['GAME_ID'] = data['pg_data']['GAME_ID'].unique()
    
    if 'pg_matchup' in data:
        matchup = data['pg_matchup'].copy()
        
        # Group by game
        game_matchup = matchup.groupby('GAME_ID')
        
        # Calculate matchup advantage metrics
        matchup_features = pd.DataFrame(index=features['GAME_ID'])
        
        if 'PLAYER_POINTS' in matchup.columns and 'MATCHUP_FIELD_GOALS_MADE' in matchup.columns:
            matchup_features['avg_matchup_advantage'] = game_matchup.apply(
                lambda x: (x['PLAYER_POINTS'] - x['MATCHUP_FIELD_GOALS_MADE'] * 2).mean()
            )
            
        if 'PLAYER_POINTS' in matchup.columns and 'MATCHUP_FIELD_GOALS_ATTEMPTED' in matchup.columns:
            matchup_features['matchup_efficiency'] = game_matchup.apply(
                lambda x: (x['PLAYER_POINTS'] / (x['MATCHUP_FIELD_GOALS_ATTEMPTED'] + 0.1)).mean()
            )
        
        # Merge matchup features
        features = features.merge(matchup_features, on='GAME_ID', how='left')
    
    return features

def create_advanced_team_metrics(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create advanced team metrics from team data."""
    features = pd.DataFrame()
    features['GAME_ID'] = data['tg_data']['GAME_ID'].unique()
    
    # Advanced team metrics
    features['pace_factor'] = data['tg_data']['PACE'] / data['tg_data']['PACE'].mean()
    features['offensive_efficiency'] = data['tg_data']['OFFENSIVE_RATING']
    features['defensive_efficiency'] = data['tg_data']['DEFENSIVE_RATING']
    features['net_rating'] = data['tg_data']['NET_RATING']
    
    # Clutch performance metrics
    clutch_data = data['tt_clutch'].set_index('TEAM_ID')
    features['clutch_rating'] = data['tg_data']['TEAM_ID'].map(
        clutch_data['PLUS_MINUS'].to_dict()
    )
    
    return features

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the feature creation
    from scripts.feature_engineering import load_processed_data
    data = load_processed_data("C:/Projects/NBA_Prediction")
    features = create_advanced_features(data)
    print(f"Created {len(features.columns)} advanced features") 