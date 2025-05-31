"""
NBA Data Processing Pipeline Runner

This script processes and merges the pre-processed NBA data files into consolidated datasets.

Column Mappings:
- All column mappings are stored in data/processed/columns/
- core_column_names.csv: Core data file mappings
- advanced_column_names.csv: Advanced stats mappings
- defense_column_names.csv: Defense stats mappings
- tracking_column_names.csv: Player tracking mappings
- matchups_betting_column_names.csv: Matchup and betting mappings
- file_name is the direct path to the files or folders with many files
- -if the path ends with a folder (example: C:/projects/NBA_prediction/data/raw/tracking/shotchartdetail/details/) then all files in the folder need to be processed
- -if the path ends with a .csv file (example: C:/projects/nba_prediction/data/raw/core/games.csv) then only that file needs to be processed
- data_type column is to describe the type of data in the data file (p = player, t = team, a = administrative, dt = dashteam dp = dashplayer, b = both)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any
import os
import shutil
from datetime import datetime
from utils.process_data import DataProcessor
import json
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NBADataPipeline:
    def __init__(self, base_dir: str = "C:/Projects/NBA_Prediction"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "data" / "raw"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.features_dir = self.processed_dir / "features"
        self.merged_dir = self.processed_dir / "merged"
        self.output_dir = self.processed_dir

        self.features_dir.mkdir(parents=True, exist_ok=True)
        Path('logs').mkdir(exist_ok=True)

        self.column_mapping = self._load_column_mapping()

    def _load_column_mapping(self) -> Dict[str, Any]:
        """Load the column mapping from FINAL_col_mapping.json."""
        mapping_path = self.processed_dir / "columns" / "FINAL_col_mapping.json"
        try:
            with open(mapping_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading column mapping: {str(e)}")
            return {}

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to a consistent format."""
        # Convert all column names to uppercase
        df.columns = df.columns.str.upper()
        
        # Standardize common column name variations
        column_mapping = {
            'PERSON_ID': 'PLAYER_ID',
            'ATHLETE_ID': 'PLAYER_ID',
            'CLOSE_DEF_PERSON_ID': 'PLAYER_ID',
            'TEAM_CITY': 'TEAM_NAME',
            'TEAM_TRICODE': 'TEAM_ABBREVIATION',
            'TEAM_SLUG': 'TEAM_ABBREVIATION',
            'GAME_DATE_TIME': 'GAME_DATE',
            'DATE': 'GAME_DATE',
            'FAMILY_NAME': 'PLAYER_LAST_NAME',
            'FIRST_NAME': 'PLAYER_FIRST_NAME',
            'NAME_I': 'PLAYER_NAME',
            'PLAYER_SLUG': 'PLAYER_NAME',
            'MATCHUP_MIN': 'MINUTES',
            'MIN': 'MINUTES',
            'MIN_SEC': 'MINUTES',
            'FGM': 'FIELD_GOALS_MADE',
            'FGA': 'FIELD_GOALS_ATTEMPTED',
            'FG_PCT': 'FIELD_GOAL_PERCENTAGE',
            'FG3M': 'THREE_POINTERS_MADE',
            'FG3A': 'THREE_POINTERS_ATTEMPTED',
            'FG3_PCT': 'THREE_POINTER_PERCENTAGE',
            'FTM': 'FREE_THROWS_MADE',
            'FTA': 'FREE_THROWS_ATTEMPTED',
            'FT_PCT': 'FREE_THROW_PERCENTAGE',
            'OREB': 'OFFENSIVE_REBOUNDS',
            'DREB': 'DEFENSIVE_REBOUNDS',
            'REB': 'REBOUNDS',
            'AST': 'ASSISTS',
            'TOV': 'TURNOVERS',
            'STL': 'STEALS',
            'BLK': 'BLOCKS',
            'PF': 'PERSONAL_FOULS',
            'PTS': 'POINTS',
            'PLUS_MINUS': 'PLUS_MINUS_POINTS'
        }
        
        df = df.rename(columns=column_mapping)
        return df

    def _standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types for common columns."""
        # Convert date columns to datetime
        date_columns = ['GAME_DATE']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
        
        # Convert numeric columns
        numeric_columns = [
            'MINUTES', 'FIELD_GOALS_MADE', 'FIELD_GOALS_ATTEMPTED', 'FIELD_GOAL_PERCENTAGE',
            'THREE_POINTERS_MADE', 'THREE_POINTERS_ATTEMPTED', 'THREE_POINTER_PERCENTAGE',
            'FREE_THROWS_MADE', 'FREE_THROWS_ATTEMPTED', 'FREE_THROW_PERCENTAGE',
            'OFFENSIVE_REBOUNDS', 'DEFENSIVE_REBOUNDS', 'REBOUNDS', 'ASSISTS',
            'TURNOVERS', 'STEALS', 'BLOCKS', 'PERSONAL_FOULS', 'POINTS',
            'PLUS_MINUS_POINTS'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def _handle_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate columns by keeping the most recent non-null value."""
        # Get duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()]
        
        for col in duplicate_cols:
            # Get all columns with this name
            cols = [c for c in df.columns if c == col]
            
            # Create a new column with the most recent non-null value
            df[col] = df[cols].apply(lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None, axis=1)
            
            # Drop the duplicate columns
            df = df.drop(columns=cols[1:])
        
        return df

    def _read_data_file(self, file_path: Path, expected_columns: List[str]) -> Optional[pd.DataFrame]:
        try:
            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return None
            
            # Read the file
            df = pd.read_csv(file_path)
            
            # Standardize column names
            df = self._standardize_column_names(df)
            
            # Standardize data types
            df = self._standardize_data_types(df)
            
            # Handle duplicate columns
            df = self._handle_duplicate_columns(df)
            
            # Filter out empty columns from expected_columns
            non_empty_expected_columns = [col for col in expected_columns if col and not pd.isna(col)]
            
            # Check for missing columns
            missing_cols = set(non_empty_expected_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing columns in {file_path}: {missing_cols}")
            
            # Select only the expected columns that exist
            available_cols = [col for col in non_empty_expected_columns if col in df.columns]
            
            # Log null value statistics for each column
            for col in available_cols:
                if col in df.columns:
                    null_count = df[col].isna().sum()
                    if null_count > 0:
                        logger.info(f"Column {col} has {null_count} null values out of {len(df)} rows")
            
            return df[available_cols]
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return None

    def _process_game_files(self, mapping: Dict[str, Any]) -> pd.DataFrame:
        dfs = []
        try:
            path = Path(str(mapping['path']).replace('\\', os.sep).replace('/', os.sep))
            if mapping['type'] == 'file':
                df = self._read_data_file(path, mapping['columns'])
                if df is not None:
                    dfs.append(df)
            else:
                for file_path in path.rglob('*.csv'):
                    df = self._read_data_file(file_path, mapping['columns'])
                    if df is not None:
                        dfs.append(df)
        except Exception as e:
            logger.error(f"Error processing {mapping['path']}: {str(e)}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def process_data_by_type(self, data_type: str) -> Dict[str, pd.DataFrame]:
        logger.info(f"Processing {data_type} data...")
        results = {key: pd.DataFrame() for key in ['p', 't', 'a', 'dt', 'dp', 'b']}
        type_mappings = {k: v for k, v in self.column_mapping.items() if v['data_type'] == data_type}
        
        for mapping in type_mappings.values():
            try:
                df = self._process_game_files(mapping)
                if not df.empty:
                    df.columns = [col.upper() for col in df.columns]
                    type_code = mapping['data_type_code']
                    
                    # Handle PLAYER_LAST_TEAM_ID in tracking data
                    if data_type == 'tracking' and 'PLAYER_LAST_TEAM_ID' in df.columns:
                        # For tracking data, we want to keep the current team's data
                        df = df[df['TEAM_ID'] == df['PLAYER_LAST_TEAM_ID']]
                        df = df.drop(columns=['PLAYER_LAST_TEAM_ID'])
                    
                    # Define merge keys based on data type and type code
                    merge_key = []
                    if type_code == 'p':
                        # Handle different player ID column names and case
                        if data_type == 'advanced':
                            # Log the current column names
                            logger.info(f"Current columns in advanced data: {list(df.columns)}")
                            
                            # Convert all columns to lowercase for advanced data
                            df.columns = [col.lower() for col in df.columns]
                            logger.info(f"Columns after lowercase conversion: {list(df.columns)}")
                            
                            # First check for person_id
                            if 'person_id' in df.columns:
                                logger.info("Found person_id column")
                                pass  # Already lowercase
                            else:
                                logger.warning(f"No person_id column found in {mapping['path']}")
                                continue
                            
                            # Ensure we have the required columns
                            required_cols = ['game_id', 'team_id', 'person_id']
                            if all(col in df.columns for col in required_cols):
                                merge_key = required_cols
                                logger.info(f"Found all required columns: {required_cols}")
                                logger.info(f"Advanced player data shape before merge: {df.shape}")
                                
                                # First, ensure we only have one row per player per game
                                df = df.drop_duplicates(subset=merge_key, keep='first')
                                logger.info(f"Shape after removing duplicates: {df.shape}")
                                
                                # Define a function to merge values, keeping non-null values
                                def merge_values(series):
                                    # For numeric columns, take the max non-null value
                                    if np.issubdtype(series.dtype, np.number):
                                        return series.max()
                                    # For string columns, take the first non-null value
                                    else:
                                        non_null = series.dropna()
                                        return non_null.iloc[0] if not non_null.empty else np.nan
                                
                                # Group by the merge key and merge all columns
                                df = df.groupby(merge_key, as_index=False).agg(merge_values)
                                logger.info(f"Advanced player data shape after merge: {df.shape}")
                                logger.info(f"Number of unique games: {df['game_id'].nunique()}")
                                logger.info(f"Number of unique players per game: {df.groupby('game_id')['person_id'].nunique().mean()}")
                                
                                # Log the number of rows per game to verify
                                rows_per_game = df.groupby('game_id').size()
                                logger.info(f"Rows per game statistics:\n{rows_per_game.describe()}")
                                
                                # If this is not the first advanced data file, merge with existing data
                                if type_code in results and not results[type_code].empty:
                                    logger.info(f"Merging with existing {data_type} {type_code} data")
                                    logger.info(f"Existing data shape: {results[type_code].shape}")
                                    
                                    # First, ensure we don't have duplicate rows in either DataFrame
                                    df = df.drop_duplicates(subset=merge_key, keep='first')
                                    results[type_code] = results[type_code].drop_duplicates(subset=merge_key, keep='first')
                                    
                                    # Merge with outer join to keep all rows
                                    results[type_code] = pd.merge(
                                        results[type_code],
                                        df,
                                        on=merge_key,
                                        how='outer',
                                        suffixes=('', '_new')
                                    )
                                    
                                    # For columns that exist in both, keep the non-null value from the new data
                                    for col in df.columns:
                                        if col not in merge_key:
                                            if col in results[type_code].columns:
                                                # If we have a _new suffix column, use it to fill nulls
                                                if f"{col}_new" in results[type_code].columns:
                                                    results[type_code][col] = results[type_code][col].fillna(results[type_code][f"{col}_new"])
                                                    results[type_code] = results[type_code].drop(columns=[f"{col}_new"])
                                    
                                    logger.info(f"Shape after merge with existing data: {results[type_code].shape}")
                                else:
                                    results[type_code] = df
                            else:
                                missing_cols = [col for col in required_cols if col not in df.columns]
                                logger.warning(f"Missing required columns in {mapping['path']}: {missing_cols}")
                                continue
                        else:
                            player_id_col = 'player_id'
                            if all(col in df.columns for col in ['game_id', 'team_id', player_id_col]):
                                merge_key = ['game_id', 'team_id', player_id_col]
                        
                        if merge_key:
                            def smart_merge(series):
                                if np.issubdtype(series.dtype, np.number):
                                    return series.max()
                                else:
                                    return series.dropna().astype(str).unique()[0] if not series.dropna().empty else np.nan
                            
                            df = df.groupby(merge_key, dropna=False).agg(smart_merge).reset_index()
                            logger.info(f"Merged {data_type}_{type_code} by {merge_key} to {len(df)} rows")
                            logger.info(f"Advanced player data shape after merge: {df.shape}")
                    elif type_code == 't':
                        if data_type == 'advanced':
                            # Convert all columns to uppercase for advanced data
                            df.columns = [col.upper() for col in df.columns]
                            # Also convert game_id and team_id to uppercase
                            if 'game_id' in df.columns:
                                df = df.rename(columns={'game_id': 'GAME_ID'})
                            if 'team_id' in df.columns:
                                df = df.rename(columns={'team_id': 'TEAM_ID'})
                        if all(col in df.columns for col in ['GAME_ID', 'TEAM_ID']):
                            merge_key = ['GAME_ID', 'TEAM_ID']
                    elif type_code == 'dp':
                        # Handle dashplayer data
                        if 'leaguedashptdefend' in str(mapping['path']):
                            # Log the current column names
                            logger.info(f"Current columns in defense data: {list(df.columns)}")
                            
                            # First rename the columns
                            if 'PLAYER_LAST_TEAM_ID' in df.columns:
                                df = df.rename(columns={'PLAYER_LAST_TEAM_ID': 'TEAM_ID'})
                                logger.info(f"Renamed PLAYER_LAST_TEAM_ID to TEAM_ID")
                            
                            if 'CLOSE_DEF_PERSON_ID' in df.columns:
                                df = df.rename(columns={'CLOSE_DEF_PERSON_ID': 'PLAYER_ID'})
                                logger.info(f"Renamed CLOSE_DEF_PERSON_ID to PLAYER_ID")
                            
                            logger.info(f"Columns after rename: {list(df.columns)}")
                            
                            # Now check for the renamed columns
                            if all(col in df.columns for col in ['TEAM_ID', 'PLAYER_ID']):
                                merge_key = ['TEAM_ID', 'PLAYER_ID']
                                logger.info(f"Defense data ready for merge with {len(df)} rows")
                            else:
                                missing_cols = [col for col in ['TEAM_ID', 'PLAYER_ID'] if col not in df.columns]
                                logger.warning(f"Missing required columns after rename in defense data: {missing_cols}")
                                continue
                        else:
                            # Regular dashplayer data
                            if all(col in df.columns for col in ['TEAM_ID', 'PLAYER_ID']):
                                merge_key = ['TEAM_ID', 'PLAYER_ID']
                        
                        if merge_key:
                            # For tracking data, we want to combine all statistics for each player
                            def smart_merge(series):
                                # For tracking data, we need to handle multiple rows per player
                                if 'tracking' in mapping['data_type']:
                                    # For tracking data, we want to:
                                    # 1. Keep all unique non-null values
                                    # 2. For numeric columns, take the most recent non-null value
                                    # 3. For categorical columns, take the most common value
                                    if pd.api.types.is_numeric_dtype(series):
                                        # For numeric columns, take the most recent non-null value
                                        non_null_values = series.dropna()
                                        if len(non_null_values) > 0:
                                            return non_null_values.iloc[-1]  # Most recent value
                                        return None
                                    else:
                                        # For categorical columns, take the most common value
                                        non_null_values = series.dropna()
                                        if len(non_null_values) > 0:
                                            return non_null_values.mode().iloc[0]
                                        return None
                                elif 'advanced' in mapping['data_type']:
                                    # For advanced stats, we want to:
                                    # 1. For numeric columns, take the average of non-null values
                                    # 2. For categorical columns, take the most common value
                                    if pd.api.types.is_numeric_dtype(series):
                                        non_null_values = series.dropna()
                                        if len(non_null_values) > 0:
                                            return non_null_values.mean()
                                        return None
                                    else:
                                        non_null_values = series.dropna()
                                        if len(non_null_values) > 0:
                                            return non_null_values.mode().iloc[0]
                                        return None
                                elif 'defense' in mapping['data_type']:
                                    # For defense stats, we want to:
                                    # 1. For numeric columns, take the sum of non-null values
                                    # 2. For categorical columns, take the most common value
                                    if pd.api.types.is_numeric_dtype(series):
                                        non_null_values = series.dropna()
                                        if len(non_null_values) > 0:
                                            return non_null_values.sum()
                                        return None
                                    else:
                                        non_null_values = series.dropna()
                                        if len(non_null_values) > 0:
                                            return non_null_values.mode().iloc[0]
                                        return None
                                else:
                                    # For core stats, we want to:
                                    # 1. For numeric columns, take the max non-null value
                                    # 2. For categorical columns, take the first non-null value
                                    if pd.api.types.is_numeric_dtype(series):
                                        non_null_values = series.dropna()
                                        if len(non_null_values) > 0:
                                            return non_null_values.max()
                                        return None
                                    else:
                                        non_null_values = series.dropna()
                                        if len(non_null_values) > 0:
                                            return non_null_values.iloc[0]
                                        return None
                            
                            # Group by the merge key and combine all statistics
                            df = df.groupby(merge_key, dropna=False).agg(smart_merge).reset_index()
                            
                            # Log the merge results
                            logger.info(f"Merged {data_type}_{type_code} by {merge_key} to {len(df)} rows")
                            logger.info(f"Number of non-null values per column after merge:")
                            for col in df.columns:
                                non_null_count = df[col].notna().sum()
                                logger.info(f"  {col}: {non_null_count}/{len(df)} ({non_null_count/len(df)*100:.1f}%)")
                            
                            if type_code in results:
                                # Merge with existing data, preserving all statistics
                                results[type_code] = pd.merge(
                                    results[type_code],
                                    df,
                                    on=merge_key,
                                    how='outer',
                                    suffixes=('', '_new')
                                )
                                
                                # For columns that exist in both, keep the non-null value from the new data
                                for col in df.columns:
                                    if col not in merge_key:
                                        if col in results[type_code].columns:
                                            # If we have a _new suffix column, use it to fill nulls
                                            if f"{col}_new" in results[type_code].columns:
                                                results[type_code][col] = results[type_code][col].fillna(results[type_code][f"{col}_new"])
                                                results[type_code] = results[type_code].drop(columns=[f"{col}_new"])
                                
                                logger.info(f"Added {len(df)} rows to {data_type} {type_code} data")
                            else:
                                results[type_code] = df
                    elif type_code == 'dt':
                        if all(col in df.columns for col in ['GAME_ID', 'TEAM_ID']):
                            merge_key = ['GAME_ID', 'TEAM_ID']
                            def smart_merge(series):
                                if np.issubdtype(series.dtype, np.number):
                                    return series.max()
                                else:
                                    return series.dropna().astype(str).unique()[0] if not series.dropna().empty else np.nan
                            
                            df = df.groupby(merge_key, dropna=False).agg(smart_merge).reset_index()
                            logger.info(f"Merged {data_type}_{type_code} by {merge_key} to {len(df)} rows")
                            
                            if type_code in results:
                                results[type_code] = pd.concat([results[type_code], df], ignore_index=True)
                                logger.info(f"Added {len(df)} rows to {data_type} {type_code} data")
            except Exception as e:
                logger.error(f"Error processing {mapping['path']}: {str(e)}")

        # Final processing for each type
        for type_code, df in results.items():
            if not df.empty:
                # Define final merge keys
                merge_key = []
                if type_code == 'p' and all(col in df.columns for col in ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']):
                    merge_key = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']
                elif type_code == 't' and all(col in df.columns for col in ['GAME_ID', 'TEAM_ID']):
                    merge_key = ['GAME_ID', 'TEAM_ID']
                elif type_code == 'dp':
                    if 'leaguedashptdefend' in str(mapping['path']):
                        # Special case for defense data
                        if all(col in df.columns for col in ['GAME_ID', 'PLAYER_ID']):
                            merge_key = ['GAME_ID', 'PLAYER_ID']
                    else:
                        # Regular dashplayer data
                        if all(col in df.columns for col in ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']):
                            merge_key = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']
                elif type_code == 'dt' and all(col in df.columns for col in ['GAME_ID', 'TEAM_ID']):
                    merge_key = ['GAME_ID', 'TEAM_ID']
                
                if merge_key:
                    def smart_merge(series):
                        if np.issubdtype(series.dtype, np.number):
                            return series.max()
                        else:
                            return series.dropna().astype(str).unique()[0] if not series.dropna().empty else np.nan
                    
                    df = df.groupby(merge_key, dropna=False).agg(smart_merge).reset_index()
                    logger.info(f"Final merge of {data_type}_{type_code} by {merge_key} to {len(df)} rows")
                
                # Sort the data appropriately
                if data_type == 'tracking':
                    if type_code == 'p' and all(col in df.columns for col in ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']):
                        df = df.sort_values(['GAME_ID', 'TEAM_ID', 'PLAYER_ID'])
                        logger.info(f"Sorted tracking p data by GAME_ID, TEAM_ID, PLAYER_ID")
                    elif type_code == 'dp':
                        if 'leaguedashptdefend' in str(mapping['path']):
                            if all(col in df.columns for col in ['GAME_ID', 'PLAYER_ID']):
                                df = df.sort_values(['GAME_ID', 'PLAYER_ID'])
                                logger.info(f"Sorted tracking dp defense data by GAME_ID, PLAYER_ID")
                        else:
                            if all(col in df.columns for col in ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']):
                                df = df.sort_values(['GAME_ID', 'TEAM_ID', 'PLAYER_ID'])
                                logger.info(f"Sorted tracking dp data by GAME_ID, TEAM_ID, PLAYER_ID")
                    elif type_code == 'dt' and all(col in df.columns for col in ['GAME_ID', 'TEAM_ID']):
                        df = df.sort_values(['GAME_ID', 'TEAM_ID'])
                        logger.info(f"Sorted tracking dt data by GAME_ID, TEAM_ID")
                elif data_type == 'advanced':
                    if type_code == 'p' and all(col in df.columns for col in ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']):
                        df = df.sort_values(['GAME_ID', 'TEAM_ID', 'PLAYER_ID'])
                        logger.info(f"Sorted advanced p data by GAME_ID, TEAM_ID, PLAYER_ID")
                    elif type_code == 't' and all(col in df.columns for col in ['GAME_ID', 'TEAM_ID']):
                        df = df.sort_values(['GAME_ID', 'TEAM_ID'])
                        logger.info(f"Sorted advanced t data by GAME_ID, TEAM_ID")
                
                output_path = self.processed_dir / f"{data_type}_{type_code}_data.csv"
                df.to_csv(output_path, index=False)
                logger.info(f"Saved {len(df)} rows of {data_type} {type_code} data to {output_path}")
        
        return results

    def validate_games_and_odds(self) -> None:
        logger.info("Validating games and odds data...")
        games = pd.read_csv(self.base_dir / "data" / "raw" / "core" / "games.csv")
        odds = pd.read_csv(self.base_dir / "data" / "raw" / "betting" / "nba_sbr_odds_2025.csv")
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        odds['date'] = pd.to_datetime(odds['date'])
        odds['game_id'] = odds['date'].dt.strftime('%Y%m%d') + '_' + odds['away_team'] + '_' + odds['home_team']
        games.to_csv(self.base_dir / "data" / "processed" / "validated_games.csv", index=False)
        odds.to_csv(self.base_dir / "data" / "processed" / "validated_odds.csv", index=False)
        logger.info(f"Saved validated games data with {len(games)} games")
        logger.info(f"Saved validated odds data with {len(odds)} games")

    def _read_and_standardize(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Read a CSV file and standardize its column names."""
        try:
            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return None
            
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.upper()
            return df
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return None

    def _merge_dataframes(self, dfs: List[pd.DataFrame], merge_keys: List[str], 
                         sort_keys: List[str] = None) -> pd.DataFrame:
        """Merge multiple dataframes on specified keys and sort if needed."""
        if not dfs:
            return pd.DataFrame()
        
        # Merge all dataframes
        merged_df = dfs[0]
        for df in dfs[1:]:
            # Log the columns we're about to merge
            logger.info(f"Columns in first DataFrame: {merged_df.columns.tolist()}")
            logger.info(f"Columns in second DataFrame: {df.columns.tolist()}")
            
            # Simple merge keeping all columns
            merged_df = pd.merge(
                merged_df,
                df,
                on=merge_keys,
                how='outer'
            )
            
            # Remove any suffixes from column names
            merged_df.columns = [col.split('_x')[0].split('_y')[0] for col in merged_df.columns]
            
            logger.info(f"Shape after merge: {merged_df.shape}")
            logger.info(f"Columns after merge: {merged_df.columns.tolist()}")
        
        # Sort if sort keys are provided
        if sort_keys:
            merged_df = merged_df.sort_values(by=sort_keys)
        
        return merged_df

    def _copy_and_rename_file(self, source_file: str, target_file: str) -> None:
        """Copy and rename a file from source to target location."""
        source_path = self.merged_dir / source_file
        target_path = self.output_dir / target_file
        
        if not source_path.exists():
            logger.warning(f"Source file does not exist: {source_path}")
            return
            
        try:
            shutil.copy2(source_path, target_path)
            logger.info(f"Copied {source_file} to {target_file}")
        except Exception as e:
            logger.error(f"Error copying {source_file} to {target_file}: {str(e)}")

    def create_matchup_data(self) -> None:
        """Create consolidated matchup data by merging home and away matchup files."""
        logger.info("Creating pg_matchup.csv...")
        
        # Read the matchup files
        home_matchup_df = pd.read_csv("data/processed/merged/advanced_pg_matchup_home.csv")
        away_matchup_df = pd.read_csv("data/processed/merged/advanced_pg_matchup_away.csv")
        
        # Log initial data shapes and column orders
        logger.info(f"Initial shapes - Home Matchup: {home_matchup_df.shape}, Away Matchup: {away_matchup_df.shape}")
        logger.info(f"Home Matchup columns: {home_matchup_df.columns.tolist()}")
        logger.info(f"Away Matchup columns: {away_matchup_df.columns.tolist()}")
        
        # Store original column order from home matchup data
        original_columns = home_matchup_df.columns.tolist()
        
        # Merge the dataframes
        merged_df = pd.concat([home_matchup_df, away_matchup_df], ignore_index=True)
        
        # Sort by game and player while maintaining original column order
        merged_df = merged_df.sort_values(['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID'])
        
        # Ensure columns are in the original order
        merged_df = merged_df[original_columns]
        
        # Log final data shape and column order
        logger.info(f"Final merged data shape: {merged_df.shape}")
        logger.info(f"Final column order: {merged_df.columns.tolist()}")
        
        # Save to output file
        output_path = "data/processed/pg_matchup.csv"
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Created pg_matchup.csv with {len(merged_df)} rows")

    def create_pg_data(self) -> None:
        """Create consolidated player game data."""
        logger.info("Creating pg_data.csv...")
        
        # Read the source files
        tracking_df = pd.read_csv("data/processed/merged/tracking_pg_gamelogs.csv")
        defense_df = pd.read_csv("data/processed/merged/defense_pg_data.csv")
        advanced_df = pd.read_csv("data/processed/merged/advanced_pg_data.csv")
        
        # Log initial data shapes and columns
        logger.info(f"Initial shapes - Tracking: {tracking_df.shape}, Defense: {defense_df.shape}, Advanced: {advanced_df.shape}")
        logger.info(f"Tracking columns: {tracking_df.columns.tolist()}")
        logger.info(f"Defense columns: {defense_df.columns.tolist()}")
        logger.info(f"Advanced columns: {advanced_df.columns.tolist()}")
        
        # Format GAME_DATE in tracking data to only show date
        tracking_df['GAME_DATE'] = pd.to_datetime(tracking_df['GAME_DATE']).dt.date
        
        # Columns to drop from advanced and defense data
        columns_to_drop = [
            'PLAYER_TEAM_NAME', 'PLAYER_TEAM_CITY', 'PLAYER_TEAM_ABBREVIATION',
            'PLAYER_TEAM_SLUG', 'PLAYER_FIRST_NAME', 'PLAYER_LAST_NAME',
            'PLAYER_INITIAL', 'PLAYER_SLUG', 'COMMENT'
        ]
        
        # Drop specified columns from defense and advanced data
        defense_df = defense_df.drop(columns=[col for col in columns_to_drop if col in defense_df.columns])
        advanced_df = advanced_df.drop(columns=[col for col in columns_to_drop if col in advanced_df.columns])
        
        # Verify merge keys exist in all dataframes
        merge_keys = ['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID']
        for df, name in [(tracking_df, 'Tracking'), (defense_df, 'Defense'), (advanced_df, 'Advanced')]:
            missing_keys = [key for key in merge_keys if key not in df.columns]
            if missing_keys:
                logger.error(f"Missing merge keys in {name} data: {missing_keys}")
                return
        
        # Merge all dataframes using the improved merge method
        dfs = [tracking_df, defense_df, advanced_df]
        merged_df = self._merge_dataframes(dfs, merge_keys, sort_keys=merge_keys)
        
        # Define the exact column order
        column_order = [
            'GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID', 'GAME_DATE', 'MATCHUP',
            'AWAY_TEAM_ID', 'HOME_TEAM_ID', 'PLAYER_TEAM_NAME', 'PLAYER_TEAM_ABBREVIATION',
            'PLAYER_NAME', 'JERSEY_NUM', 'POSITION'
        ]
        
        # Add remaining columns that aren't in the specified order
        remaining_cols = [col for col in merged_df.columns if col not in column_order]
        final_column_order = column_order + remaining_cols
        
        # Reorder columns
        merged_df = merged_df[final_column_order]
        
        # Filter out only completely empty rows (no player information)
        # Keep rows that have at least the basic player identification
        has_player_info = merged_df[['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID']].notna().all(axis=1)
        rows_before = len(merged_df)
        merged_df = merged_df[has_player_info]
        rows_removed = rows_before - len(merged_df)
        logger.info(f"Removed {rows_removed} rows with no player identification")
        logger.info(f"Final shape after filtering: {merged_df.shape}")
        
        # Log some statistics about the data
        logger.info("Data statistics:")
        logger.info(f"Total games: {merged_df['GAME_ID'].nunique()}")
        logger.info(f"Total players: {merged_df['PLAYER_ID'].nunique()}")
        logger.info(f"Average players per game: {len(merged_df) / merged_df['GAME_ID'].nunique():.2f}")
        
        # Save to output file
        output_path = "data/processed/pg_data.csv"
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Created pg_data.csv with {len(merged_df)} rows")
        logger.info(f"Final columns: {merged_df.columns.tolist()}")

    def create_tg_data(self) -> None:
        """Create consolidated team game data."""
        logger.info("Creating tg_data.csv...")
        
        # Read source files
        files = [
            "advanced_tg_data.csv",
            "defense_tg_data.csv"
        ]
        
        dfs = []
        for file in files:
            df = self._read_and_standardize(self.merged_dir / file)
            if df is not None:
                dfs.append(df)
                logger.info(f"Loaded {file} with shape {df.shape}")
                logger.info(f"Columns in {file}: {df.columns.tolist()}")
        
        # Merge dataframes
        merge_keys = ["GAME_ID", "TEAM_ID"]
        sort_keys = ["GAME_ID", "TEAM_ID"]
        
        # Use a more robust merge strategy
        merged_df = None
        for df in dfs:
            if merged_df is None:
                merged_df = df.copy()
            else:
                # Merge with existing data, keeping all columns
                merged_df = pd.merge(
                    merged_df,
                    df,
                    on=merge_keys,
                    how='outer',
                    suffixes=('', '_y')
                )
                
                # Remove duplicate columns (those ending with _y)
                duplicate_cols = [col for col in merged_df.columns if col.endswith('_y')]
                if duplicate_cols:
                    logger.info(f"Removing duplicate columns: {duplicate_cols}")
                    merged_df = merged_df.drop(columns=duplicate_cols)
        
        if merged_df is not None:
            # Sort the data
            merged_df = merged_df.sort_values(sort_keys)
            
            # Fill any missing values with 0 for numeric columns
            numeric_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns
            merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
            
            # Save to output
            output_path = self.output_dir / "tg_data.csv"
            merged_df.to_csv(output_path, index=False)
            logger.info(f"Created tg_data.csv with {len(merged_df)} rows")
            logger.info(f"Final columns: {merged_df.columns.tolist()}")
        else:
            logger.error("No data was merged for tg_data.csv")

    def create_pt_data(self) -> None:
        """Create consolidated player total data."""
        logger.info("Creating pt_data.csv...")
        
        # Read source files
        files = [
            "dashboard_pt_data.csv",
            "defense_pt_data.csv"
        ]
        
        dfs = []
        for file in files:
            df = self._read_and_standardize(self.merged_dir / file)
            if df is not None:
                dfs.append(df)
                logger.info(f"Loaded {file} with shape {df.shape}")
        
        # Merge dataframes
        merge_keys = ["PLAYER_TEAM_ID", "PLAYER_ID"]
        sort_keys = ["PLAYER_TEAM_ID", "PLAYER_ID"]
        
        merged_df = self._merge_dataframes(dfs, merge_keys, sort_keys)
        
        # Save to output
        output_path = self.output_dir / "pt_data.csv"
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Created pt_data.csv with {len(merged_df)} rows")
        logger.info(f"Final columns: {merged_df.columns.tolist()}")

    def create_tt_data(self) -> None:
        """Create consolidated team total data."""
        logger.info("Creating tt_data.csv...")
        
        # Read source files
        files = [
            "dashboard_tt_teamstats.csv",
            "defense_tt_data.csv"
        ]
        
        dfs = []
        for file in files:
            df = self._read_and_standardize(self.merged_dir / file)
            if df is not None:
                dfs.append(df)
                logger.info(f"Loaded {file} with shape {df.shape}")
        
        # Merge dataframes
        merge_keys = ["TEAM_ID"]
        sort_keys = ["TEAM_ID"]
        
        merged_df = self._merge_dataframes(dfs, merge_keys, sort_keys)
        
        # Save to output
        output_path = self.output_dir / "tt_data.csv"
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Created tt_data.csv with {len(merged_df)} rows")
        logger.info(f"Final columns: {merged_df.columns.tolist()}")

    def copy_standalone_files(self) -> None:
        """Copy and rename standalone files to the processed directory."""
        file_mappings = {
            "advanced_tg_starter_data.csv": "tg_starter.csv",
            "advanced_tg_bench_data.csv": "tg_bench.csv",
            "core_pt_matchupsrollup_data.csv": "pt_matchupsrollup.csv",
            "core_pg_pbp_data.csv": "pg_pbp.csv",
            "validated_games.csv": "validated_games.csv",
            "validated_odds.csv": "validated_odds.csv",
            "dashboard_tt_clutch.csv": "tt_clutch.csv",
            "tracking_pg_shotchart_leagueaverages.csv": "pg_shotchart_averages.csv",
            "tracking_pg_shotchartdetail.csv": "pg_shotchart_detail.csv"
        }
        
        for source, target in file_mappings.items():
            self._copy_and_rename_file(source, target)

    def run(self):
        """Run the complete data processing pipeline."""
        try:
            logger.info("Starting data processing pipeline...")
            
            # Create consolidated files
            self.create_pg_data()
            self.create_matchup_data()
            self.create_tg_data()
            self.create_pt_data()
            self.create_tt_data()
            
            # Copy standalone files
            self.copy_standalone_files()
            
            logger.info("Data processing pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = NBADataPipeline()
    pipeline.run()
