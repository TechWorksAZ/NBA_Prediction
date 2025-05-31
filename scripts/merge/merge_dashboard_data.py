import pandas as pd
import os
from pathlib import Path
import logging
from typing import List, Dict, Set, Any
import numpy as np
from tqdm import tqdm
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardDataMerger:
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
            
    def _process_team_clutch(self) -> None:
        """Process and save team clutch data."""
        logger.info("Processing team clutch data...")
        
        input_path = self.raw_dir / "dashboard" / "leaguedashteamclutch_2024-25.csv"
        output_path = self.processed_dir / "merged" / "dashboard_tt_clutch.csv"
        
        try:
            # Read the data
            df = pd.read_csv(input_path)
            
            # Sort by TEAM_ID
            df = df.sort_values('TEAM_ID')
            
            # Save the processed data
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Team clutch data saved to {output_path}")
            logger.info(f"Columns in team clutch data: {list(df.columns)}")
            
        except Exception as e:
            logger.error(f"Error processing team clutch data: {str(e)}")
            raise
            
    def _process_team_stats(self) -> None:
        """Process and save team stats data."""
        logger.info("Processing team stats data...")
        
        input_path = self.raw_dir / "dashboard" / "leaguedashteamstats_2024-25.csv"
        output_path = self.processed_dir / "merged" / "dashboard_tt_teamstats.csv"
        
        try:
            # Read the data
            df = pd.read_csv(input_path)
            
            # Sort by TEAM_ID
            df = df.sort_values('TEAM_ID')
            
            # Save the processed data
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Team stats data saved to {output_path}")
            logger.info(f"Columns in team stats data: {list(df.columns)}")
            
        except Exception as e:
            logger.error(f"Error processing team stats data: {str(e)}")
            raise
            
    def _process_player_data(self) -> None:
        """Process and merge all player dashboard data."""
        logger.info("Processing and merging player dashboard data...")
        
        # Define input paths
        ptstats_path = self.raw_dir / "dashboard" / "leaguedashptstats_2024-25.csv"
        ptdefend_path = self.raw_dir / "dashboard" / "leaguedashptdefend_2024-25.csv"
        ptplayer_path = self.raw_dir / "dashboard" / "leaguedashplayerstats_2024-25.csv"
        output_path = self.processed_dir / "merged" / "dashboard_pt_data.csv"
        
        try:
            # Read all data files
            ptstats_df = pd.read_csv(ptstats_path)
            ptdefend_df = pd.read_csv(ptdefend_path)
            ptplayer_df = pd.read_csv(ptplayer_path)
            
            # Rename columns in ptdefend_df
            ptdefend_df = ptdefend_df.rename(columns={
                'CLOSE_DEF_PERSON_ID': 'PLAYER_ID',
                'PLAYER_LAST_TEAM_ID': 'PLAYER_TEAM_ID',
                'PLAYER_LAST_TEAM_ABBREVIATION': 'PLAYER_TEAM_ABBREVIATION'
            })
            
            # Rename columns in ptstats_df and ptplayer_df
            for df in [ptstats_df, ptplayer_df]:
                df.rename(columns={
                    'TEAM_ID': 'PLAYER_TEAM_ID',
                    'TEAM_ABBREVIATION': 'PLAYER_TEAM_ABBREVIATION'
                }, inplace=True)
            
            # Merge all dataframes
            # First merge ptstats and ptplayer
            merged_df = pd.merge(
                ptstats_df,
                ptplayer_df,
                on=['PLAYER_ID', 'PLAYER_TEAM_ID', 'PLAYER_TEAM_ABBREVIATION'],
                how='outer',
                suffixes=('_ptstats', '_ptplayer')
            )
            
            # Then merge with ptdefend
            merged_df = pd.merge(
                merged_df,
                ptdefend_df,
                on=['PLAYER_ID', 'PLAYER_TEAM_ID', 'PLAYER_TEAM_ABBREVIATION'],
                how='outer',
                suffixes=('', '_ptdefend')
            )
            
            # Sort by PLAYER_TEAM_ID then PLAYER_ID
            merged_df = merged_df.sort_values(['PLAYER_TEAM_ID', 'PLAYER_ID'])
            
            # Save the merged data
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_csv(output_path, index=False)
            logger.info(f"Player dashboard data saved to {output_path}")
            logger.info(f"Columns in merged player data: {list(merged_df.columns)}")
            
        except Exception as e:
            logger.error(f"Error processing player dashboard data: {str(e)}")
            raise
            
    def merge_dashboard_data(self) -> None:
        """Merge all dashboard data sources."""
        try:
            self._process_team_clutch()
            self._process_team_stats()
            self._process_player_data()
            logger.info("Dashboard data merge completed successfully")
            
        except Exception as e:
            logger.error(f"Error in dashboard data merge: {str(e)}")
            raise

def main():
    try:
        merger = DashboardDataMerger()
        merger.merge_dashboard_data()
        
    except Exception as e:
        logger.error(f"Error in dashboard data merge: {str(e)}")
        raise

if __name__ == "__main__":
    main() 