import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedDataMerger:
    def __init__(self):
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.advanced_dir = self.raw_dir / "advanced"
        self.output_dir = self.processed_dir / "merged"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load column mappings
        self.mappings = self._load_column_mappings()
        
    def _load_column_mappings(self) -> Dict:
        """Load column mappings from JSON file."""
        mapping_file = self.processed_dir / "columns" / "FINAL_col_mapping.json"
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
        return mappings.get("advanced", {})

    def _read_and_combine_directory(self, directory: Path) -> pd.DataFrame:
        """Read and combine all CSV files in a directory."""
        if not directory.exists():
            logging.warning(f"Directory does not exist: {directory}")
            return pd.DataFrame()
            
        all_files = list(directory.glob("*.csv"))
        if not all_files:
            logging.warning(f"No CSV files found in directory: {directory}")
            return pd.DataFrame()
        
        logging.info(f"Found {len(all_files)} CSV files in {directory}")
        for file in all_files:
            logging.info(f"Processing file: {file}")
        
        dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                logging.info(f"Successfully read file: {file} with {len(df)} rows")
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error reading {file}: {e}")
                continue
        
        if not dfs:
            logging.warning(f"No data frames created from files in {directory}")
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Combined {len(dfs)} files with total {len(df)} rows from {directory}")
        return df

    def _standardize_player_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for player data."""
        # Convert all columns to uppercase first
        df.columns = [col.upper() for col in df.columns]
        
        # Define column mappings for player data
        column_mapping = {
            'GAME_ID': 'GAME_ID',
            'TEAM_ID': 'PLAYER_TEAM_ID',
            'TEAM_NAME': 'PLAYER_TEAM_NAME',
            'TEAM_CITY': 'PLAYER_TEAM_CITY',
            'TEAM_TRICODE': 'PLAYER_TEAM_ABBREVIATION',
            'TEAM_SLUG': 'PLAYER_TEAM_SLUG',
            'PERSON_ID': 'PLAYER_ID',
            'FIRST_NAME': 'PLAYER_FIRST_NAME',
            'FAMILY_NAME': 'PLAYER_LAST_NAME',
            'NAME_I': 'PLAYER_INITIAL',
            'PLAYER_SLUG': 'PLAYER_SLUG',
            'POSITION': 'POSITION',
            'COMMENT': 'COMMENT',
            'JERSEY_NUM': 'JERSEY_NUM'
        }
        
        df = df.rename(columns=column_mapping)
        return df

    def _standardize_team_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for team data."""
        # Convert all columns to uppercase first
        df.columns = [col.upper() for col in df.columns]
        
        # Define column mappings for team data
        column_mapping = {
            'GAME_ID': 'GAME_ID',
            'TEAM_ID': 'TEAM_ID',
            'TEAM_NAME': 'TEAM_NAME',
            'TEAM_CITY': 'TEAM_CITY',
            'TEAM_TRICODE': 'TEAM_ABBREVIATION',
            'TEAM_SLUG': 'TEAM_SLUG'
        }
        
        df = df.rename(columns=column_mapping)
        return df

    def _merge_matchup_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Merge matchup data from boxscorematchupsv3 into separate home and away files."""
        home_dfs = []
        away_dfs = []
        
        # Process matchup data specifically
        for data_type, sources in self.mappings.items():
            for source_name, source_data in sources.items():
                if source_data.get("type") == "pg" and "boxscorematchupsv3" in str(source_data["path"]):
                    # Log the mapping details
                    logging.info(f"Processing matchup data for source: {source_name}")
                    logging.info(f"Source data type: {data_type}")
                    logging.info(f"Original path from mapping: {source_data['path']}")
                    
                    # Get home and away player data paths
                    base_path = str(source_data["path"])
                    if "home_player" in base_path:
                        home_dir = Path(base_path)
                        home_data = self._read_and_combine_directory(home_dir)
                        if not home_data.empty:
                            home_data = self._standardize_player_columns(home_data)
                            logging.info(f"Home matchup data from {home_dir}: {len(home_data)} rows")
                            logging.info(f"Unique GAME_ID, PLAYER_TEAM_ID, PLAYER_ID combinations: {home_data[['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID']].drop_duplicates().shape[0]}")
                            home_dfs.append(home_data)
                    elif "away_player" in base_path:
                        away_dir = Path(base_path)
                        away_data = self._read_and_combine_directory(away_dir)
                        if not away_data.empty:
                            away_data = self._standardize_player_columns(away_data)
                            logging.info(f"Away matchup data from {away_dir}: {len(away_data)} rows")
                            logging.info(f"Unique GAME_ID, PLAYER_TEAM_ID, PLAYER_ID combinations: {away_data[['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID']].drop_duplicates().shape[0]}")
                            away_dfs.append(away_data)
        
        # Merge home matchup data
        home_matchup_data = pd.DataFrame()
        if home_dfs:
            home_matchup_data = pd.concat(home_dfs, ignore_index=True)
            # Remove any duplicates that might have been introduced during concatenation
            home_matchup_data = home_matchup_data.drop_duplicates()
            home_matchup_data = home_matchup_data.sort_values(['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID'])
            logging.info(f"Final home matchup data shape: {home_matchup_data.shape}")
            logging.info(f"Number of unique games in home matchup data: {home_matchup_data['GAME_ID'].nunique()}")
        else:
            logging.warning("No home matchup data was found!")
        
        # Merge away matchup data
        away_matchup_data = pd.DataFrame()
        if away_dfs:
            away_matchup_data = pd.concat(away_dfs, ignore_index=True)
            # Remove any duplicates that might have been introduced during concatenation
            away_matchup_data = away_matchup_data.drop_duplicates()
            away_matchup_data = away_matchup_data.sort_values(['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID'])
            logging.info(f"Final away matchup data shape: {away_matchup_data.shape}")
            logging.info(f"Number of unique games in away matchup data: {away_matchup_data['GAME_ID'].nunique()}")
        else:
            logging.warning("No away matchup data was found!")
        
        return home_matchup_data, away_matchup_data

    def _merge_player_data(self) -> pd.DataFrame:
        """Merge all player game data from all advanced sources."""
        player_dfs = []
        
        # Process each advanced data type
        for data_type, sources in self.mappings.items():
            for source_name, source_data in sources.items():
                if source_data.get("type") == "pg" and "boxscorematchupsv3" not in str(source_data["path"]):
                    # Get home and away player data
                    home_dir = Path(source_data["path"])
                    away_dir = Path(source_data["path"].replace("home_player", "away_player"))
                    
                    home_data = self._read_and_combine_directory(home_dir)
                    away_data = self._read_and_combine_directory(away_dir)
                    
                    if not home_data.empty:
                        home_data = self._standardize_player_columns(home_data)
                        logging.info(f"Player data from {home_dir}: {len(home_data)} rows")
                        logging.info(f"Unique GAME_ID, PLAYER_TEAM_ID, PLAYER_ID combinations: {home_data[['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID']].drop_duplicates().shape[0]}")
                        player_dfs.append(home_data)
                    if not away_data.empty:
                        away_data = self._standardize_player_columns(away_data)
                        logging.info(f"Player data from {away_dir}: {len(away_data)} rows")
                        logging.info(f"Unique GAME_ID, PLAYER_TEAM_ID, PLAYER_ID combinations: {away_data[['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID']].drop_duplicates().shape[0]}")
                        player_dfs.append(away_data)
        
        if player_dfs:
            all_player_data = pd.concat(player_dfs, ignore_index=True)
            # Check for duplicates before grouping
            duplicates = all_player_data.duplicated(subset=['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID'], keep=False)
            if duplicates.any():
                logging.warning(f"Found {duplicates.sum()} duplicate rows in player data. These will be handled by taking the first occurrence.")
                all_player_data = all_player_data.groupby(['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID']).first().reset_index()
            else:
                logging.info("No duplicates found in player data, skipping groupby")
            
            all_player_data = all_player_data.sort_values(['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID'])
            logging.info(f"Final player data shape: {all_player_data.shape}")
            logging.info(f"Number of unique games in player data: {all_player_data['GAME_ID'].nunique()}")
            return all_player_data
        return pd.DataFrame()

    def _merge_team_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Merge all team game data from all advanced sources."""
        team_dfs = []
        starter_dfs = []
        bench_dfs = []
        
        # Process each advanced data type
        for data_type, sources in self.mappings.items():
            for source_name, source_data in sources.items():
                if source_data.get("type") == "tg":
                    # Get home and away team data
                    home_dir = Path(source_data["path"])
                    away_dir = Path(source_data["path"].replace("home_totals", "away_totals"))
                    
                    home_data = self._read_and_combine_directory(home_dir)
                    away_data = self._read_and_combine_directory(away_dir)
                    
                    if not home_data.empty:
                        home_data = self._standardize_team_columns(home_data)
                        logging.info(f"Team data from {home_dir}: {len(home_data)} rows")
                        logging.info(f"Unique GAME_ID, TEAM_ID combinations: {home_data[['GAME_ID', 'TEAM_ID']].drop_duplicates().shape[0]}")
                        team_dfs.append(home_data)
                    if not away_data.empty:
                        away_data = self._standardize_team_columns(away_data)
                        logging.info(f"Team data from {away_dir}: {len(away_data)} rows")
                        logging.info(f"Unique GAME_ID, TEAM_ID combinations: {away_data[['GAME_ID', 'TEAM_ID']].drop_duplicates().shape[0]}")
                        team_dfs.append(away_data)
                    
                    # Handle traditional boxscore starter/bench data
                    if data_type == "boxscoretraditionalv3":
                        home_starter_dir = Path(source_data["path"].replace("home_totals", "home_starter_totals"))
                        away_starter_dir = Path(source_data["path"].replace("home_totals", "away_starter_totals"))
                        home_bench_dir = Path(source_data["path"].replace("home_totals", "home_bench_totals"))
                        away_bench_dir = Path(source_data["path"].replace("home_totals", "away_bench_totals"))
                        
                        # Process starter data
                        home_starter = self._read_and_combine_directory(home_starter_dir)
                        away_starter = self._read_and_combine_directory(away_starter_dir)
                        if not home_starter.empty:
                            home_starter = self._standardize_team_columns(home_starter)
                            logging.info(f"Starter data from {home_starter_dir}: {len(home_starter)} rows")
                            starter_dfs.append(home_starter)
                        if not away_starter.empty:
                            away_starter = self._standardize_team_columns(away_starter)
                            logging.info(f"Starter data from {away_starter_dir}: {len(away_starter)} rows")
                            starter_dfs.append(away_starter)
                        
                        # Process bench data
                        home_bench = self._read_and_combine_directory(home_bench_dir)
                        away_bench = self._read_and_combine_directory(away_bench_dir)
                        if not home_bench.empty:
                            home_bench = self._standardize_team_columns(home_bench)
                            logging.info(f"Bench data from {home_bench_dir}: {len(home_bench)} rows")
                            bench_dfs.append(home_bench)
                        if not away_bench.empty:
                            away_bench = self._standardize_team_columns(away_bench)
                            logging.info(f"Bench data from {away_bench_dir}: {len(away_bench)} rows")
                            bench_dfs.append(away_bench)
        
        # Merge regular team data
        all_team_data = pd.DataFrame()
        if team_dfs:
            all_team_data = pd.concat(team_dfs, ignore_index=True)
            # Check for duplicates before grouping
            duplicates = all_team_data.duplicated(subset=['GAME_ID', 'TEAM_ID'], keep=False)
            if duplicates.any():
                logging.warning(f"Found {duplicates.sum()} duplicate rows in team data. These will be handled by taking the first occurrence.")
                all_team_data = all_team_data.groupby(['GAME_ID', 'TEAM_ID']).first().reset_index()
            else:
                logging.info("No duplicates found in team data, skipping groupby")
            
            all_team_data = all_team_data.sort_values(['GAME_ID', 'TEAM_ID'])
            logging.info(f"Final team data shape: {all_team_data.shape}")
            logging.info(f"Number of unique games in team data: {all_team_data['GAME_ID'].nunique()}")
            
        # Merge starter data
        all_starter_data = pd.DataFrame()
        if starter_dfs:
            all_starter_data = pd.concat(starter_dfs, ignore_index=True)
            # Check for duplicates before grouping
            duplicates = all_starter_data.duplicated(subset=['GAME_ID', 'TEAM_ID'], keep=False)
            if duplicates.any():
                logging.warning(f"Found {duplicates.sum()} duplicate rows in starter data. These will be handled by taking the first occurrence.")
                all_starter_data = all_starter_data.groupby(['GAME_ID', 'TEAM_ID']).first().reset_index()
            else:
                logging.info("No duplicates found in starter data, skipping groupby")
            
            all_starter_data = all_starter_data.sort_values(['GAME_ID', 'TEAM_ID'])
            logging.info(f"Final starter data shape: {all_starter_data.shape}")
            logging.info(f"Number of unique games in starter data: {all_starter_data['GAME_ID'].nunique()}")
            
        # Merge bench data
        all_bench_data = pd.DataFrame()
        if bench_dfs:
            all_bench_data = pd.concat(bench_dfs, ignore_index=True)
            # Check for duplicates before grouping
            duplicates = all_bench_data.duplicated(subset=['GAME_ID', 'TEAM_ID'], keep=False)
            if duplicates.any():
                logging.warning(f"Found {duplicates.sum()} duplicate rows in bench data. These will be handled by taking the first occurrence.")
                all_bench_data = all_bench_data.groupby(['GAME_ID', 'TEAM_ID']).first().reset_index()
            else:
                logging.info("No duplicates found in bench data, skipping groupby")
            
            all_bench_data = all_bench_data.sort_values(['GAME_ID', 'TEAM_ID'])
            logging.info(f"Final bench data shape: {all_bench_data.shape}")
            logging.info(f"Number of unique games in bench data: {all_bench_data['GAME_ID'].nunique()}")
            
        return all_team_data, all_starter_data, all_bench_data

    def merge_advanced_data(self):
        """Main method to merge all advanced data."""
        try:
            # Merge player game data (pg)
            player_data = self._merge_player_data()
            if not player_data.empty:
                output_path = self.output_dir / "advanced_pg_data.csv"
                player_data.to_csv(output_path, index=False)
                logging.info(f"Saved merged player game data to {output_path}")
            
            # Merge matchup data
            home_matchup_data, away_matchup_data = self._merge_matchup_data()
            if not home_matchup_data.empty:
                output_path = self.output_dir / "advanced_pg_matchup_home.csv"
                home_matchup_data.to_csv(output_path, index=False)
                logging.info(f"Saved home matchup data to {output_path}")
            if not away_matchup_data.empty:
                output_path = self.output_dir / "advanced_pg_matchup_away.csv"
                away_matchup_data.to_csv(output_path, index=False)
                logging.info(f"Saved away matchup data to {output_path}")
            
            # Merge team game data (tg)
            team_data, starter_data, bench_data = self._merge_team_data()
            
            if not team_data.empty:
                output_path = self.output_dir / "advanced_tg_data.csv"
                team_data.to_csv(output_path, index=False)
                logging.info(f"Saved merged team game data to {output_path}")
                
            if not starter_data.empty:
                output_path = self.output_dir / "advanced_tg_starter_data.csv"
                starter_data.to_csv(output_path, index=False)
                logging.info(f"Saved merged team starter data to {output_path}")
                
            if not bench_data.empty:
                output_path = self.output_dir / "advanced_tg_bench_data.csv"
                bench_data.to_csv(output_path, index=False)
                logging.info(f"Saved merged team bench data to {output_path}")
                
            logging.info("Advanced data merge completed successfully")
            
        except Exception as e:
            logging.error(f"Error merging advanced data: {e}")
            raise

def main():
    """Main function to merge advanced data and save results."""
    print("Starting advanced data merge...")
    
    # Create output directory if it doesn't exist
    output_dir = Path('data/processed/merged')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the merger
    merger = AdvancedDataMerger()
    
    # Merge the data
    merger.merge_advanced_data()
    
    print("Advanced data merge complete!")

if __name__ == "__main__":
    main() 