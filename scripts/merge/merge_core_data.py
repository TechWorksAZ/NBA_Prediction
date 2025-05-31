import pandas as pd
import os
from pathlib import Path
import logging
from typing import List, Dict, Set, Any
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import ast
import pytz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_json_column(value):
    """Clean a JSON-like column value."""
    if pd.isna(value):
        return None
    try:
        # Try to parse as JSON first
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            # If that fails, try to parse as Python literal
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If both fail, return the original value
            return value

def convert_game_time(time_str):
    """Convert game time from UTC to local time and extract only the time part."""
    if pd.isna(time_str):
        return None
    try:
        # Parse the UTC time
        utc_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        # Convert to local time (assuming Eastern time for NBA games)
        local_time = utc_time.astimezone(pytz.timezone('US/Eastern'))
        # Return only the time part in HH:MM format
        return local_time.strftime('%H:%M')
    except Exception as e:
        logger.warning(f"Error converting game time {time_str}: {str(e)}")
        return None

class CoreDataMerger:
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.column_mappings = self._load_column_mappings()
        
    def _load_column_mappings(self) -> Dict[str, Any]:
        """Load column mappings from JSON file."""
        mappings_path = self.processed_dir / "columns" / "FINAL_col_mapping.json"
        try:
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
                # Only return the core data mappings
                return mappings.get("core", {})
        except FileNotFoundError:
            logger.error(f"Column mappings file not found at {mappings_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in mappings file at {mappings_path}")
            raise
            
    def _standardize_date(self, date_str: str) -> str:
        """Convert various date formats to a standard format."""
        try:
            # Try parsing as ISO format first
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            try:
                # Try parsing as other common formats
                for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
            except Exception:
                pass
        return date_str  # Return original if all parsing fails
            
    def _merge_games_and_schedule(self) -> None:
        """Merge games.csv and nba_schedule.csv based on date and game ID."""
        logger.info("Merging games and schedule data...")
        
        # Read games data
        games_path = self.raw_dir / "core" / "games.csv"
        games_df = pd.read_csv(games_path)
        
        # Read schedule data
        schedule_path = self.raw_dir / "core" / "nba_schedule.csv"
        schedule_df = pd.read_csv(schedule_path)
        
        # Convert all column names to uppercase in schedule data
        schedule_df.columns = schedule_df.columns.str.upper()
        
        # Standardize dates in both dataframes
        games_df['GAME_DATE'] = games_df['GAME_DATE'].apply(self._standardize_date)
        schedule_df['GAME_DATE'] = schedule_df['GAME_DATE'].apply(self._standardize_date)
        
        # Convert game time to local time
        if 'GAME_DATE_TIME_EST' in schedule_df.columns:
            schedule_df['GAME_TIME'] = schedule_df['GAME_DATE_TIME_EST'].apply(convert_game_time)
        
        # List of time-related columns to exclude
        time_columns_to_exclude = [
            'GAME_DATE_EST', 'GAME_TIME_EST', 'GAME_DATE_TIME_EST',
            'GAME_DATE_UTC', 'GAME_TIME_UTC', 'GAME_DATE_TIME_UTC',
            'AWAY_TEAM_TIME', 'HOME_TEAM_TIME'
        ]
        
        # Get all columns except the time-related ones
        schedule_columns = [col for col in schedule_df.columns if col not in time_columns_to_exclude]
        schedule_df = schedule_df[schedule_columns]
        
        # Merge on date and game ID
        merged_df = pd.merge(
            games_df,
            schedule_df,
            on=['GAME_DATE', 'GAME_ID'],
            how='outer'
        )
        
        # Rename columns to match our standard format
        column_mappings = {
            # Game Info
            'GAME_CODE': 'GAME_CODE',
            'GAME_STATUS': 'GAME_STATUS',
            'GAME_STATUS_TEXT': 'GAME_STATUS_TEXT',
            'GAME_SEQUENCE': 'GAME_SEQUENCE',
            'DAY': 'DAY',
            'MONTH_NUM': 'MONTH_NUM',
            'WEEK_NUMBER': 'WEEK_NUMBER',
            'WEEK_NAME': 'WEEK_NAME',
            'IF_NECESSARY': 'IF_NECESSARY',
            'SERIES_GAME_NUMBER': 'SERIES_GAME_NUMBER',
            'GAME_LABEL': 'GAME_LABEL',
            'GAME_SUB_LABEL': 'GAME_SUB_LABEL',
            'SERIES_TEXT': 'SERIES_TEXT',
            
            # Venue Info
            'ARENA_NAME': 'VENUE_NAME',
            'ARENA_STATE': 'VENUE_STATE',
            'ARENA_CITY': 'VENUE_CITY',
            'POSTPONED_STATUS': 'POSTPONED_STATUS',
            'BRANCH_LINK': 'BRANCH_LINK',
            'GAME_SUBTYPE': 'GAME_SUBTYPE',
            'IS_NEUTRAL': 'NEUTRAL_SITE',
            
            # Home Team Info
            'HOME_TEAM_CITY': 'HOME_TEAM_CITY',
            'HOME_TEAM_TRICODE': 'HOME_TEAM_ABBR',
            'HOME_TEAM_SLUG': 'HOME_TEAM_SLUG',
            'HOME_TEAM_WINS': 'HOME_TEAM_WINS',
            'HOME_TEAM_LOSSES': 'HOME_TEAM_LOSSES',
            'HOME_TEAM_SCORE': 'HOME_SCORE',
            'HOME_TEAM_SEED': 'HOME_TEAM_SEED',
            
            # Away Team Info
            'AWAY_TEAM_CITY': 'AWAY_TEAM_CITY',
            'AWAY_TEAM_TRICODE': 'AWAY_TEAM_ABBR',
            'AWAY_TEAM_SLUG': 'AWAY_TEAM_SLUG',
            'AWAY_TEAM_WINS': 'AWAY_TEAM_WINS',
            'AWAY_TEAM_LOSSES': 'AWAY_TEAM_LOSSES',
            'AWAY_TEAM_SCORE': 'AWAY_SCORE',
            'AWAY_TEAM_SEED': 'AWAY_TEAM_SEED',
            
            # Season Info
            'SEASON': 'SEASON',
            'LEAGUE_ID': 'LEAGUE_ID',
            'SEASON_TYPE_ID': 'SEASON_TYPE_ID',
            'SEASON_TYPE_DESCRIPTION': 'SEASON_TYPE'
        }
        
        # Apply column mappings
        merged_df = merged_df.rename(columns=column_mappings)
        
        # Select final columns in desired order
        final_columns = [
            # Game Identification
            'GAME_ID', 'GAME_CODE', 'GAME_DATE', 'GAME_TIME', 'SEASON', 'SEASON_TYPE',
            'SEASON_TYPE_ID', 'SEASON_TYPE_DESCRIPTION',
            
            # Game Status
            'GAME_STATUS', 'GAME_STATUS_TEXT', 'GAME_SEQUENCE', 'POSTPONED_STATUS',
            'GAME_LABEL', 'GAME_SUB_LABEL', 'SERIES_TEXT', 'SERIES_GAME_NUMBER',
            'IF_NECESSARY',
            
            # Date Info
            'DAY', 'MONTH_NUM', 'WEEK_NUMBER', 'WEEK_NAME',
            
            # Home Team Info
            'HOME_TEAM_ID', 'HOME_TEAM_ABBR', 'HOME_TEAM_NAME', 'HOME_TEAM_CITY',
            'HOME_TEAM_SLUG', 'HOME_TEAM_WINS', 'HOME_TEAM_LOSSES', 'HOME_SCORE',
            'HOME_TEAM_SEED',
            
            # Away Team Info
            'AWAY_TEAM_ID', 'AWAY_TEAM_ABBR', 'AWAY_TEAM_NAME', 'AWAY_TEAM_CITY',
            'AWAY_TEAM_SLUG', 'AWAY_TEAM_WINS', 'AWAY_TEAM_LOSSES', 'AWAY_SCORE',
            'AWAY_TEAM_SEED',
            
            # Game Details
            'MATCHUP', 'NEUTRAL_SITE', 'GAME_SUBTYPE', 'BRANCH_LINK',
            
            # Venue Info
            'VENUE_NAME', 'VENUE_CITY', 'VENUE_STATE'
        ]
        
        # Filter to only columns that exist
        final_columns = [col for col in final_columns if col in merged_df.columns]
        merged_df = merged_df[final_columns]
        
        # Save merged data
        output_path = self.processed_dir / "merged" / "validated_games.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Merged games and schedule data saved to {output_path}")
        logger.info(f"Columns in merged games data: {list(merged_df.columns)}")
        
    def _merge_play_by_play(self) -> None:
        """Merge all play-by-play files from the pbp directory."""
        logger.info("Merging play-by-play data...")
        
        pbp_dir = self.raw_dir / "core" / "pbp"
        all_pbp_data = []
        
        # Read all CSV files in the pbp directory
        for file in pbp_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                all_pbp_data.append(df)
            except Exception as e:
                logger.warning(f"Error reading {file}: {str(e)}")
        
        if not all_pbp_data:
            logger.error("No play-by-play data found")
            return
            
        # Combine all data
        pbp_df = pd.concat(all_pbp_data, ignore_index=True)
        
        # Standardize player-related column names
        column_mappings = {
            'player1_id': 'PLAYER_ID',
            'player1_name': 'PLAYER_NAME',
            'player1_team_city': 'PLAYER_TEAM_CITY',
            'player1_team_id': 'PLAYER_TEAM_ID',
            'player1_team_nickname': 'PLAYER_TEAM_NICKNAME',
            'player1_team_abbreviation': 'PLAYER_TEAM_ABBREVIATION',
            'player2_id': 'PLAYER2_ID',
            'player2_name': 'PLAYER2_NAME',
            'player2_team_city': 'PLAYER2_TEAM_CITY',
            'player2_team_id': 'PLAYER2_TEAM_ID',
            'player2_team_nickname': 'PLAYER2_TEAM_NICKNAME',
            'player2_team_abbreviation': 'PLAYER2_TEAM_ABBREVIATION',
            'player3_id': 'PLAYER3_ID',
            'player3_name': 'PLAYER3_NAME',
            'player3_team_city': 'PLAYER3_TEAM_CITY',
            'player3_team_id': 'PLAYER3_TEAM_ID',
            'player3_team_nickname': 'PLAYER3_TEAM_NICKNAME',
            'player3_team_abbreviation': 'PLAYER3_TEAM_ABBREVIATION'
        }
        
        # Rename columns
        pbp_df = pbp_df.rename(columns=column_mappings)
        
        # Capitalize all remaining column names
        pbp_df.columns = [col.upper() for col in pbp_df.columns]
        
        # Sort by GAME_ID and event_num to maintain chronological order
        pbp_df = pbp_df.sort_values(['GAME_ID', 'EVENT_NUM'])
        
        # Save merged data
        output_path = self.processed_dir / "merged" / "core_pg_pbp_data.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pbp_df.to_csv(output_path, index=False)
        logger.info(f"Merged play-by-play data saved to {output_path}")
        logger.info(f"Columns in merged PBP data: {list(pbp_df.columns)}")
        
    def _process_matchups_rollup(self) -> None:
        """Process matchups rollup data with column name changes."""
        logger.info("Processing matchups rollup data...")
        
        matchups_path = self.raw_dir / "core" / "matchupsrollup" / "2025.csv"
        try:
            matchups_df = pd.read_csv(matchups_path)
            
            # Rename columns
            matchups_df = matchups_df.rename(columns={
                'DEF_PLAYER_ID': 'PLAYER_ID',
                'DEF_PLAYER_NAME': 'PLAYER_NAME'
            })
            
            # Capitalize all column names
            matchups_df.columns = [col.upper() for col in matchups_df.columns]
            
            # Save processed data
            output_path = self.processed_dir / "merged" / "core_pg_matchupsrollup_data.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            matchups_df.to_csv(output_path, index=False)
            logger.info(f"Processed matchups rollup data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing matchups rollup data: {str(e)}")
            raise
            
    def merge_core_data(self) -> None:
        """Merge all core data sources."""
        try:
            self._merge_games_and_schedule()
            self._merge_play_by_play()
            self._process_matchups_rollup()
            logger.info("Core data merge completed successfully")
            
        except Exception as e:
            logger.error(f"Error in core data merge: {str(e)}")
            raise

def process_line_scores(line_scores_str, prefix):
    """Process line scores into a list of period scores."""
    if pd.isna(line_scores_str):
        return pd.Series([None] * 4)  # Return 4 None values for 4 periods
    
    logger.debug(f"Processing {prefix} line scores: {repr(line_scores_str)}")
    
    try:
        # First try to parse as Python literal
        try:
            # Replace newlines with commas and clean up any double commas
            line_scores_str = line_scores_str.replace('\n', ',').replace(',,', ',')
            # Add commas between objects if missing
            line_scores_str = line_scores_str.replace('} {', '}, {')
            line_scores = ast.literal_eval(str(line_scores_str))
            logger.debug(f"Successfully parsed {prefix} line scores as Python literal: {line_scores}")
        except (ValueError, SyntaxError) as e:
            logger.debug(f"Failed to parse as Python literal: {e}")
            # If that fails, try to clean and parse as JSON
            line_scores_str = str(line_scores_str)
            # Remove any single quotes and replace with double quotes
            line_scores_str = line_scores_str.replace("'", '"')
            # Replace newlines with commas
            line_scores_str = line_scores_str.replace('\n', ',')
            # Add commas between objects if missing
            line_scores_str = line_scores_str.replace('} {', '}, {')
            # Clean up any double commas
            line_scores_str = line_scores_str.replace(',,', ',')
            # Wrap in brackets if not already
            if not line_scores_str.startswith('['):
                line_scores_str = f"[{line_scores_str}]"
            logger.debug(f"Cleaned line scores string: {line_scores_str}")
            # Try parsing as JSON
            line_scores = json.loads(line_scores_str)
            logger.debug(f"Successfully parsed {prefix} line scores as JSON: {line_scores}")
        
        if not isinstance(line_scores, list):
            logger.warning(f"Invalid line scores format for {prefix}: {line_scores_str}")
            return pd.Series([None] * 4)
        
        # Extract period scores
        period_scores = [None] * 4  # Initialize with None values
        for i, period in enumerate(line_scores):
            if i < 4 and isinstance(period, dict) and 'value' in period:  # Only process first 4 periods
                period_scores[i] = period['value']
        
        logger.debug(f"Extracted period scores for {prefix}: {period_scores}")
        return pd.Series(period_scores)
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError, SyntaxError) as e:
        logger.warning(f"Error processing line scores for {prefix}: {str(e)}")
        return pd.Series([None] * 4)

def process_records(records_str, prefix):
    """Process records into a single record string."""
    if pd.isna(records_str):
        return None
    
    logger.debug(f"Processing {prefix} records: {repr(records_str)}")
    
    try:
        # First try to parse as Python literal
        try:
            # Replace newlines with commas and clean up any double commas
            records_str = records_str.replace('\n', ',').replace(',,', ',')
            records = ast.literal_eval(str(records_str))
            logger.debug(f"Successfully parsed {prefix} records as Python literal: {records}")
        except (ValueError, SyntaxError) as e:
            logger.debug(f"Failed to parse as Python literal: {e}")
            # If that fails, try to clean and parse as JSON
            records_str = str(records_str)
            # Remove any single quotes and replace with double quotes
            records_str = records_str.replace("'", '"')
            # Replace newlines with commas
            records_str = records_str.replace('\n', ',')
            # Add commas between objects if missing
            records_str = records_str.replace('} {', '}, {')
            # Clean up any double commas
            records_str = records_str.replace(',,', ',')
            # Wrap in brackets if not already
            if not records_str.startswith('['):
                records_str = f"[{records_str}]"
            logger.debug(f"Cleaned records string: {records_str}")
            # Try parsing as JSON
            records = json.loads(records_str)
            logger.debug(f"Successfully parsed {prefix} records as JSON: {records}")
        
        if not isinstance(records, list):
            logger.warning(f"Invalid records format for {prefix}: {records_str}")
            return None
        
        # Find the overall record
        for record in records:
            if isinstance(record, dict) and record.get('type') == 'total':
                summary = record.get('summary')
                logger.debug(f"Found record for {prefix}: {summary}")
                return summary
        logger.warning(f"No total record found for {prefix}")
        return None
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError, SyntaxError) as e:
        logger.warning(f"Error processing records for {prefix}: {str(e)}")
        return None

def process_core_data():
    """Process and clean the core NBA data."""
    logger.info("Processing core NBA data...")
    
    # Read the NBA schedule data
    schedule_path = Path("data/raw/core/nba_schedule.csv")
    if not schedule_path.exists():
        logger.error(f"Schedule file not found: {schedule_path}")
        return
        
    df = pd.read_csv(schedule_path)
    
    # Clean JSON-like columns
    json_columns = [
        'home_linescores',
        'home_records',
        'away_linescores',
        'away_records',
        'broadcast',
        'highlights',
        'notes_type',
        'notes_headline',
        'broadcast_market',
        'broadcast_name'
    ]
    
    for col in json_columns:
        if col in df.columns:
            logger.info(f"Cleaning column: {col}")
            df[col] = df[col].apply(clean_json_column)
    
    # Rename columns to match our standard format
    column_mapping = {
        'id': 'GAME_ID',
        'date': 'GAME_DATE',
        'time_valid': 'GAME_TIME',
        'neutral_site': 'NEUTRAL_SITE',
        'conference_competition': 'CONFERENCE_GAME',
        'play_by_play_available': 'PBP_AVAILABLE',
        'recent': 'RECENT_GAME',
        'venue_full_name': 'VENUE_NAME',
        'venue_address_city': 'VENUE_CITY',
        'venue_indoor': 'VENUE_INDOOR',
        'format_regulation_periods': 'REGULATION_PERIODS',
        'home_id': 'HOME_TEAM_ID',
        'home_name': 'HOME_TEAM_NAME',
        'home_abbreviation': 'HOME_TEAM_ABBR',
        'home_score': 'HOME_SCORE',
        'home_winner': 'HOME_WINNER',
        'away_id': 'AWAY_TEAM_ID',
        'away_name': 'AWAY_TEAM_NAME',
        'away_abbreviation': 'AWAY_TEAM_ABBR',
        'away_score': 'AWAY_SCORE',
        'away_winner': 'AWAY_WINNER'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Select and order columns
    columns = [
        'GAME_ID', 'GAME_DATE', 'GAME_TIME', 'SEASON', 'SEASON_TYPE',
        'GAME_STATUS', 'GAME_STATE', 'GAME_COMPLETED', 'GAME_DESCRIPTION',
        'GAME_DETAIL', 'HOME_TEAM_ID', 'HOME_TEAM_ABBR', 'HOME_TEAM_NAME',
        'HOME_SCORE', 'HOME_WINNER', 'HOME_LINESCORES', 'HOME_RECORDS',
        'AWAY_TEAM_ID', 'AWAY_TEAM_ABBR', 'AWAY_TEAM_NAME', 'AWAY_SCORE',
        'AWAY_WINNER', 'AWAY_LINESCORES', 'AWAY_RECORDS', 'MATCHUP',
        'CONFERENCE_GAME', 'NEUTRAL_SITE', 'REGULATION_PERIODS',
        'VENUE_NAME', 'VENUE_CITY', 'VENUE_STATE', 'VENUE_INDOOR',
        'ATTENDANCE', 'PBP_AVAILABLE', 'RECENT_GAME'
    ]
    
    # Keep only the columns we want
    df = df[[col for col in columns if col in df.columns]]
    
    # Save the cleaned data
    output_path = Path("data/processed/validated_games.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned games data to {output_path}")
    logger.info(f"Processed {len(df)} games")

def main():
    """Main function to merge core data."""
    logger.info("Starting core data merge...")
    
    # Load schedule data
    logger.info("Loading schedule data...")
    schedule_df = pd.read_csv('data/raw/core/nba_schedule.csv')
    
    try:
        merger = CoreDataMerger()
        merger.merge_core_data()
        
    except Exception as e:
        logger.error(f"Error in core data merge: {str(e)}")
        raise

if __name__ == "__main__":
    main() 