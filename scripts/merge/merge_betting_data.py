import pandas as pd
import os
from pathlib import Path
import logging
from typing import List, Dict, Set, Any
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pytz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BettingDataMerger:
    def __init__(self, base_dir: str = "C:/Projects/NBA_Prediction"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "data" / "raw"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.scripts_dir = self.base_dir / "scripts"
        self.process_dir = self.scripts_dir / "process"
        self.team_mappings = self._load_team_mappings()
        
    def _load_team_mappings(self) -> pd.DataFrame:
        """Load team mappings from the process directory."""
        try:
            team_mappings_path = Path("data/processed/utilities/team_mappings.csv")
            if not team_mappings_path.exists():
                logger.error(f"Team mappings file not found at {team_mappings_path}!")
                return pd.DataFrame()
            
            team_mappings = pd.read_csv(team_mappings_path)
            logger.info(f"Loaded team mappings with {len(team_mappings)} teams")
            return team_mappings
        except Exception as e:
            logger.error(f"Error loading team mappings: {str(e)}")
            return pd.DataFrame()
        
    def _normalize_time(self, time_str: str) -> str:
        """Convert various time formats to EST time."""
        try:
            # Try parsing as ISO format first
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            
            # Convert to EST
            est = pytz.timezone('US/Eastern')
            dt_est = dt.astimezone(est)
            
            # Return just the time in HH:MM format
            return dt_est.strftime('%H:%M')
        except ValueError:
            try:
                # Try parsing as other common formats
                for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                    try:
                        dt = datetime.strptime(time_str, fmt)
                        dt_est = dt.astimezone(pytz.timezone('US/Eastern'))
                        return dt_est.strftime('%H:%M')
                    except ValueError:
                        continue
            except Exception:
                pass
        return time_str  # Return original if all parsing fails
            
    def _standardize_team_names(self, team_name: str) -> str:
        """Standardize team names using the centralized team mappings."""
        if self.team_mappings.empty:
            logger.error("No team mappings available for standardization")
            return team_name.strip()
            
        # Create mapping dictionaries for different team name formats
        city_map = dict(zip(self.team_mappings['team_city'], self.team_mappings['FULL_TEAM_NAME']))
        name_map = dict(zip(self.team_mappings['team_name'], self.team_mappings['FULL_TEAM_NAME']))
        abbrev_map = dict(zip(self.team_mappings['team_abbrev'], self.team_mappings['FULL_TEAM_NAME']))
        
        # Try to match the team name using different formats
        team_name = team_name.strip()
        
        # First try exact match with full team name
        if team_name in self.team_mappings['FULL_TEAM_NAME'].values:
            return team_name
            
        # Try matching with city
        if team_name in city_map:
            return city_map[team_name]
            
        # Try matching with team name
        if team_name in name_map:
            return name_map[team_name]
            
        # Try matching with abbreviation
        if team_name in abbrev_map:
            return abbrev_map[team_name]
            
        # Special case for LA teams
        if team_name == "LA":
            return "Los Angeles Lakers"  # Default to Lakers if just "LA"
            
        logger.warning(f"Could not standardize team name: {team_name}")
        return team_name
            
    def merge_betting_data(self) -> None:
        """Merge all betting data files and standardize columns."""
        logger.info("Merging betting data...")
        
        betting_dir = self.raw_dir / "betting" / "sbr_daily"
        all_betting_data = []
        
        # Read all CSV files in the betting directory
        for file in betting_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                all_betting_data.append(df)
            except Exception as e:
                logger.warning(f"Error reading {file}: {str(e)}")
        
        if not all_betting_data:
            logger.error("No betting data found")
            return
            
        # Combine all data
        betting_df = pd.concat(all_betting_data, ignore_index=True)
        
        # Standardize column names
        column_mappings = {
            'game_id': 'GAME_ID',
            'date': 'GAME_DATE',
            'start_time': 'GAME_TIME',
            'away_team': 'AWAY_TEAM',
            'home_team': 'HOME_TEAM',
            'venue': 'VENUE',
            'city': 'CITY',
            'state': 'STATE',
            'status': 'STATUS',
            'away_score': 'AWAY_SCORE',
            'home_score': 'HOME_SCORE'
        }
        
        # Rename columns
        betting_df = betting_df.rename(columns=column_mappings)
        
        # Normalize time format
        betting_df['GAME_TIME'] = betting_df['GAME_TIME'].apply(self._normalize_time)
        
        # Standardize team names
        betting_df['AWAY_TEAM'] = betting_df['AWAY_TEAM'].apply(self._standardize_team_names)
        betting_df['HOME_TEAM'] = betting_df['HOME_TEAM'].apply(self._standardize_team_names)
        
        # Add team IDs
        if not self.team_mappings.empty:
            full_name_map = dict(zip(self.team_mappings['FULL_TEAM_NAME'], self.team_mappings['team_id']))
            betting_df['AWAY_TEAM_ID'] = betting_df['AWAY_TEAM'].map(full_name_map)
            betting_df['HOME_TEAM_ID'] = betting_df['HOME_TEAM'].map(full_name_map)
        
        # Capitalize all remaining column names
        betting_df.columns = [col.upper() for col in betting_df.columns]
        
        # Save merged data
        output_path = self.processed_dir / "merged" / "validated_odds.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        betting_df.to_csv(output_path, index=False)
        logger.info(f"Merged betting data saved to {output_path}")
        logger.info(f"Columns in merged betting data: {list(betting_df.columns)}")
        logger.info(f"Total rows: {len(betting_df)}")

def main():
    try:
        merger = BettingDataMerger()
        merger.merge_betting_data()
        logger.info("Betting data merge completed successfully")
        
    except Exception as e:
        logger.error(f"Error in betting data merge: {str(e)}")
        raise

if __name__ == "__main__":
    main() 