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
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        
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
        """Standardize team names to match our other data."""
        team_mappings = {
            # Eastern Conference
            "Boston": "Boston Celtics",
            "Brooklyn": "Brooklyn Nets",
            "New York": "New York Knicks",
            "Philadelphia": "Philadelphia 76ers",
            "Toronto": "Toronto Raptors",
            "Chicago": "Chicago Bulls",
            "Cleveland": "Cleveland Cavaliers",
            "Detroit": "Detroit Pistons",
            "Indiana": "Indiana Pacers",
            "Milwaukee": "Milwaukee Bucks",
            "Atlanta": "Atlanta Hawks",
            "Charlotte": "Charlotte Hornets",
            "Miami": "Miami Heat",
            "Orlando": "Orlando Magic",
            "Washington": "Washington Wizards",
            
            # Western Conference
            "Denver": "Denver Nuggets",
            "Minnesota": "Minnesota Timberwolves",
            "Oklahoma City": "Oklahoma City Thunder",
            "Portland": "Portland Trail Blazers",
            "Utah": "Utah Jazz",
            "Golden State": "Golden State Warriors",
            "LA Clippers": "Los Angeles Clippers",
            "LA Lakers": "Los Angeles Lakers",
            "Los Angeles": "Los Angeles Lakers",  # Default to Lakers if just "Los Angeles"
            "Phoenix": "Phoenix Suns",
            "Sacramento": "Sacramento Kings",
            "Dallas": "Dallas Mavericks",
            "Houston": "Houston Rockets",
            "Memphis": "Memphis Grizzlies",
            "New Orleans": "New Orleans Pelicans",
            "San Antonio": "San Antonio Spurs"
        }
        return team_mappings.get(team_name.strip(), team_name.strip())
            
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