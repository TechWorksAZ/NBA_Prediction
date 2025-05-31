import pandas as pd
import os
from pathlib import Path
import logging
from typing import List, Dict, Set, Any
import numpy as np
from tqdm import tqdm
import re
import json
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrackingDataMerger:
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self._load_column_mappings()
        
    def _load_column_mappings(self) -> None:
        """Load column mappings from FINAL_col_mapping.json."""
        try:
            with open("data/processed/columns/FINAL_col_mapping.json", "r") as f:
                self.mappings = json.load(f)
        except Exception as e:
            logger.error(f"Error loading column mappings: {str(e)}")
            raise
            
    def _extract_game_id(self, filename: str) -> str:
        """Extract game ID from filename."""
        # Example filename: "0022300001_shotchartdetail.csv" or "league_averages_0022400002.csv"
        match = re.search(r'league_averages_(\d{10})', filename)
        if match:
            return match.group(1)
        # Fallback for other patterns
        match = re.search(r'(\d{10})', filename)
        if match:
            return match.group(1)
        logger.warning(f"Could not extract game ID from filename: {filename}")
        return None
        
    def _merge_shot_chart_detail(self) -> None:
        """Merge all shot chart detail files and add GAME_ID from filename."""
        logger.info("Merging shot chart detail data...")
        
        shot_chart_dir = self.raw_dir / "tracking" / "shotchartdetail" / "details"
        all_shot_data = []
        
        # Get expected columns from mappings
        expected_columns = self.mappings["tracking"]["shotchartdetail"]["columns"]
        # Remove GRID_TYPE from expected columns if it exists
        if 'GRID_TYPE' in expected_columns:
            expected_columns.remove('GRID_TYPE')
        
        # Read all CSV files in the shot chart directory
        for file in shot_chart_dir.glob("*.csv"):
            try:
                game_id = self._extract_game_id(file.name)
                if not game_id:
                    logger.warning(f"Could not extract game ID from filename: {file.name}")
                    continue
                    
                df = pd.read_csv(file)
                df['GAME_ID'] = game_id
                
                # Drop GRID_TYPE column if it exists
                if 'GRID_TYPE' in df.columns:
                    df = df.drop('GRID_TYPE', axis=1)
                
                # Ensure all expected columns are present
                for col in expected_columns:
                    if col not in df.columns:
                        df[col] = None
                        
                # Select only expected columns
                df = df[expected_columns]
                all_shot_data.append(df)
            except Exception as e:
                logger.warning(f"Error reading {file}: {str(e)}")
        
        if not all_shot_data:
            logger.error("No shot chart detail data found")
            return
            
        # Combine all data
        shot_df = pd.concat(all_shot_data, ignore_index=True)
        
        # Capitalize all column names
        shot_df.columns = [col.upper() for col in shot_df.columns]
        
        # Save merged data
        output_path = self.processed_dir / "merged" / "tracking_pg_shotchartdetail.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shot_df.to_csv(output_path, index=False)
        logger.info(f"Merged shot chart detail data saved to {output_path}")
        logger.info(f"Columns in merged shot chart data: {list(shot_df.columns)}")
        
    def _merge_shot_chart_league_averages(self) -> None:
        """Merge all shot chart league averages files and add GAME_ID from filename."""
        logger.info("Merging shot chart league averages data...")
        
        league_avg_dir = self.raw_dir / "tracking" / "shotchartdetail" / "league_averages"
        all_league_avg_data = []
        
        # Get expected columns from mappings and add GAME_ID
        expected_columns = self.mappings["tracking"]["shotchartdetail_leagueaverages"]["columns"]
        # Remove GRID_TYPE from expected columns if it exists
        if 'GRID_TYPE' in expected_columns:
            expected_columns.remove('GRID_TYPE')
        expected_columns.insert(0, 'GAME_ID')  # Add GAME_ID as first column
        
        # Read all CSV files in the league averages directory
        for file in league_avg_dir.glob("*.csv"):
            try:
                logger.info(f"Processing file: {file.name}")
                game_id = self._extract_game_id(file.name)
                logger.info(f"Extracted GAME_ID: {game_id}")
                
                if not game_id:
                    logger.warning(f"Could not extract game ID from filename: {file.name}")
                    continue
                    
                df = pd.read_csv(file)
                logger.info(f"Original columns: {list(df.columns)}")
                
                # Drop GRID_TYPE column if it exists
                if 'GRID_TYPE' in df.columns:
                    df = df.drop('GRID_TYPE', axis=1)
                
                # Add GAME_ID column
                df['GAME_ID'] = game_id
                logger.info(f"Columns after adding GAME_ID: {list(df.columns)}")
                
                # Ensure all expected columns are present
                for col in expected_columns:
                    if col not in df.columns:
                        df[col] = None
                        
                # Select only expected columns
                df = df[expected_columns]
                logger.info(f"Columns after selecting expected columns: {list(df.columns)}")
                
                all_league_avg_data.append(df)
            except Exception as e:
                logger.warning(f"Error reading {file}: {str(e)}")
        
        if not all_league_avg_data:
            logger.error("No shot chart league averages data found")
            return
            
        # Combine all data
        league_avg_df = pd.concat(all_league_avg_data, ignore_index=True)
        
        # Capitalize all column names
        league_avg_df.columns = [col.upper() for col in league_avg_df.columns]
        logger.info(f"Final columns before saving: {list(league_avg_df.columns)}")
        
        # Save merged data
        output_path = self.processed_dir / "merged" / "tracking_pg_shotchart_leagueaverages.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        league_avg_df.to_csv(output_path, index=False)
        logger.info(f"Merged shot chart league averages data saved to {output_path}")
        logger.info(f"Columns in merged league averages data: {list(league_avg_df.columns)}")
        
    def _merge_player_game_logs(self) -> None:
        """Merge all player game logs into a single dataframe."""
        all_game_logs_data = []
        
        # Process each player's game logs
        for file_path in glob.glob(os.path.join(self.raw_dir, 'tracking', 'playergamelogs', '*.csv')):
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Original columns in {file_path}: {list(df.columns)}")
                
                # Rename team columns to player team columns
                column_mapping = {
                    'TEAM_ID': 'PLAYER_TEAM_ID',
                    'TEAM_NAME': 'PLAYER_TEAM_NAME',
                    'TEAM_ABBREVIATION': 'PLAYER_TEAM_ABBREVIATION',
                    'team_id': 'PLAYER_TEAM_ID',  # Add lowercase version
                    'team_name': 'PLAYER_TEAM_NAME',
                    'team_abbreviation': 'PLAYER_TEAM_ABBREVIATION'
                }
                
                # Rename columns that exist
                rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
                if rename_dict:
                    df = df.rename(columns=rename_dict)
                    logger.info(f"Renamed columns in {file_path}: {rename_dict}")
                
                logger.info(f"Columns after renaming in {file_path}: {list(df.columns)}")
                all_game_logs_data.append(df)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Combine all data
        game_logs_df = pd.concat(all_game_logs_data, ignore_index=True)
        
        # Define the desired column order
        desired_columns = [
            'GAME_ID',
            'PLAYER_TEAM_ID',
            'PLAYER_ID',
            'PLAYER_NAME',
            'PLAYER_TEAM_NAME',
            'PLAYER_TEAM_ABBREVIATION',
            'GAME_DATE',
            'MATCHUP',
            'WL',
            'MIN',
            'FGM',
            'FGA',
            'FG_PCT',
            'FG3M',
            'FG3A',
            'FG3_PCT',
            'FTM',
            'FTA',
            'FT_PCT',
            'OREB',
            'DREB',
            'REB',
            'AST',
            'TOV',
            'STL',
            'BLK',
            'BLKA',
            'PF',
            'PFD',
            'PTS',
            'PLUS_MINUS',
            'NBA_FANTASY_PTS',
            'DD2',
            'TD3',
            'WNBA_FANTASY_PTS'
        ]
        
        # Ensure all desired columns exist
        for col in desired_columns:
            if col not in game_logs_df.columns:
                game_logs_df[col] = None
                logger.warning(f"Added missing column {col} to final dataframe")
        
        # Reorder columns
        game_logs_df = game_logs_df[desired_columns]
        
        # Sort by key columns
        game_logs_df = game_logs_df.sort_values(['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID'])
        
        # Log sample of sorted data
        logger.info("Sample of sorted data (first 5 rows):")
        logger.info(game_logs_df[['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID']].head().to_string())
        
        # Save to processed directory
        output_path = os.path.join(self.processed_dir, 'merged', 'tracking_pg_gamelogs.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        game_logs_df.to_csv(output_path, index=False)
        logger.info(f"Saved merged tracking data to {output_path}")
        
    def merge_tracking_data(self) -> None:
        """Merge all tracking data sources."""
        try:
            self._merge_shot_chart_detail()
            self._merge_shot_chart_league_averages()
            self._merge_player_game_logs()
            logger.info("Tracking data merge completed successfully")
            
        except Exception as e:
            logger.error(f"Error in tracking data merge: {str(e)}")
            raise

def main():
    try:
        merger = TrackingDataMerger()
        merger.merge_tracking_data()
        
    except Exception as e:
        logger.error(f"Error in tracking data merge: {str(e)}")
        raise

if __name__ == "__main__":
    main() 