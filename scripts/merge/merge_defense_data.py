import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DefenseDataMerger:
    def __init__(self):
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.defense_dir = self.raw_dir / "defense"
        self.output_dir = self.processed_dir / "merged"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load column mappings
        self.mappings = self._load_column_mappings()
        
    def _load_column_mappings(self) -> Dict:
        """Load column mappings from JSON file."""
        mapping_file = self.processed_dir / "columns" / "FINAL_col_mapping.json"
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
        return mappings.get("defense", {})

    def _read_and_combine_directory(self, directory: Path) -> pd.DataFrame:
        """Read and combine all CSV files in a directory."""
        if not directory.exists():
            logging.warning(f"Directory does not exist: {directory}")
            return pd.DataFrame()
            
        all_files = list(directory.glob("*.csv"))
        if not all_files:
            logging.warning(f"No CSV files found in directory: {directory}")
            return pd.DataFrame()
        
        dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                logging.info(f"Read file: {file}")
            except Exception as e:
                logging.error(f"Error reading {file}: {e}")
                continue
        
        if not dfs:
            logging.warning(f"No data frames created from files in {directory}")
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Combined {len(dfs)} files with total {len(df)} rows from {directory}")
        return df

    def _standardize_boxscore_defensive_columns(self, df: pd.DataFrame, is_player: bool) -> pd.DataFrame:
        """Standardize column names for boxscore defensive data."""
        # Convert all columns to uppercase first
        df.columns = [col.upper() for col in df.columns]
        
        if is_player:
            # Player data column mappings
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
                'JERSEY_NUM': 'JERSEY_NUM',
                'MATCHUP_MINUTES': 'MINUTES',
                'PARTIAL_POSSESSIONS': 'PARTIAL_POSSESSIONS',
                'SWITCHES_ON': 'SWITCHES_ON',
                'PLAYER_POINTS': 'PTS',
                'DEFENSIVE_REBOUNDS': 'DREB',
                'MATCHUP_ASSISTS': 'AST',
                'MATCHUP_TURNOVERS': 'TOV',
                'STEALS': 'STL',
                'BLOCKS': 'BLK',
                'MATCHUP_FGM': 'FGM',
                'MATCHUP_FGA': 'FGA',
                'MATCHUP_FG_PCT': 'FG_PCT',
                'MATCHUP_FG3M': 'FG3M',
                'MATCHUP_FG3A': 'FG3A',
                'MATCHUP_FG3_PCT': 'FG3_PCT'
            }
        else:
            # Team data column mappings
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

    def _merge_boxscore_defensive(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Merge boxscore defensive data for players and teams."""
        # Player data (pg)
        home_player_dir = self.defense_dir / "boxscoredefensivev2" / "home_player"
        away_player_dir = self.defense_dir / "boxscoredefensivev2" / "away_player"
        
        home_player = self._read_and_combine_directory(home_player_dir)
        away_player = self._read_and_combine_directory(away_player_dir)
        
        if not home_player.empty:
            home_player = self._standardize_boxscore_defensive_columns(home_player, True)
        if not away_player.empty:
            away_player = self._standardize_boxscore_defensive_columns(away_player, True)
            
        player_df = pd.concat([home_player, away_player], ignore_index=True)
        
        # Team data (tg)
        home_team_dir = self.defense_dir / "boxscoredefensivev2" / "home_totals"
        away_team_dir = self.defense_dir / "boxscoredefensivev2" / "away_totals"
        
        home_team = self._read_and_combine_directory(home_team_dir)
        away_team = self._read_and_combine_directory(away_team_dir)
        
        if not home_team.empty:
            home_team = self._standardize_boxscore_defensive_columns(home_team, False)
        if not away_team.empty:
            away_team = self._standardize_boxscore_defensive_columns(away_team, False)
            
        team_df = pd.concat([home_team, away_team], ignore_index=True)
        
        return player_df, team_df

    def _standardize_hustle_stats_columns(self, df: pd.DataFrame, is_player: bool) -> pd.DataFrame:
        """Standardize column names for hustle stats data."""
        # Convert all columns to uppercase first
        df.columns = [col.upper() for col in df.columns]
        
        if is_player:
            # Player data column mappings
            column_mapping = {
                'TEAM_ID': 'PLAYER_TEAM_ID',
                'TEAM_NAME': 'PLAYER_TEAM_NAME',
                'TEAM_CITY': 'PLAYER_TEAM_CITY',
                'TEAM_TRICODE': 'PLAYER_TEAM_ABBREVIATION',
                'TEAM_SLUG': 'PLAYER_TEAM_SLUG'
            }
        else:
            # Team data column mappings
            column_mapping = {
                'TEAM_TRICODE': 'TEAM_ABBREVIATION'
            }
        
        df = df.rename(columns=column_mapping)
        return df

    def _merge_hustle_stats(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Merge hustle stats data for players and teams."""
        # Hustle stats boxscore
        hsb_player_dir = self.defense_dir / "hustlestatsboxscore" / "player"
        hsb_team_dir = self.defense_dir / "hustlestatsboxscore" / "team"
        
        hsb_player = self._read_and_combine_directory(hsb_player_dir)
        hsb_team = self._read_and_combine_directory(hsb_team_dir)
        
        if not hsb_player.empty:
            hsb_player = self._standardize_hustle_stats_columns(hsb_player, True)
        if not hsb_team.empty:
            hsb_team = self._standardize_hustle_stats_columns(hsb_team, False)
        
        return hsb_player, hsb_team

    def _merge_league_hustle_stats(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Merge league hustle stats for players and teams."""
        # League hustle stats
        lhs_player = pd.read_csv(self.defense_dir / "leaguehustlestatsplayer" / "2025.csv")
        lhs_team = pd.read_csv(self.defense_dir / "leaguehustlestatsteam" / "2025.csv")
        
        if not lhs_player.empty:
            lhs_player = self._standardize_hustle_stats_columns(lhs_player, True)
        if not lhs_team.empty:
            lhs_team = self._standardize_hustle_stats_columns(lhs_team, False)
        
        return lhs_player, lhs_team

    def merge_defense_data(self):
        """Main method to merge all defense data."""
        try:
            # Merge boxscore defensive data
            bd_player, bd_team = self._merge_boxscore_defensive()
            
            # Merge hustle stats data
            hsb_player, hsb_team = self._merge_hustle_stats()
            
            # Merge league hustle stats
            lhs_player, lhs_team = self._merge_league_hustle_stats()
            
            # 1. Player Game Data (pg)
            player_game_dfs = []
            if not bd_player.empty:
                player_game_dfs.append(bd_player)
            if not hsb_player.empty:
                player_game_dfs.append(hsb_player)
                
            if player_game_dfs:
                all_player_game_data = pd.concat(player_game_dfs, ignore_index=True)
                all_player_game_data = all_player_game_data.sort_values(['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID'])
                all_player_game_data = all_player_game_data.groupby(['GAME_ID', 'PLAYER_TEAM_ID', 'PLAYER_ID']).first().reset_index()
                output_path = self.output_dir / "defense_pg_data.csv"
                all_player_game_data.to_csv(output_path, index=False)
                logging.info(f"Saved merged player game data to {output_path}")
            
            # 2. Player Total Data (pt)
            if not lhs_player.empty:
                output_path = self.output_dir / "defense_pt_data.csv"
                lhs_player.to_csv(output_path, index=False)
                logging.info(f"Saved player total data to {output_path}")
            
            # 3. Team Game Data (tg)
            team_game_dfs = []
            if not bd_team.empty:
                team_game_dfs.append(bd_team)
            if not hsb_team.empty:
                team_game_dfs.append(hsb_team)
                
            if team_game_dfs:
                all_team_game_data = pd.concat(team_game_dfs, ignore_index=True)
                all_team_game_data = all_team_game_data.sort_values(['GAME_ID', 'TEAM_ID'])
                all_team_game_data = all_team_game_data.groupby(['GAME_ID', 'TEAM_ID']).first().reset_index()
                output_path = self.output_dir / "defense_tg_data.csv"
                all_team_game_data.to_csv(output_path, index=False)
                logging.info(f"Saved merged team game data to {output_path}")
            
            # 4. Team Total Data (tt)
            if not lhs_team.empty:
                output_path = self.output_dir / "defense_tt_data.csv"
                lhs_team.to_csv(output_path, index=False)
                logging.info(f"Saved team total data to {output_path}")
                
            logging.info("Defense data merge completed successfully")
            
        except Exception as e:
            logging.error(f"Error merging defense data: {e}")
            raise

def main():
    """Main function to merge defense data and save results."""
    print("Starting defense data merge...")
    
    # Create output directory if it doesn't exist
    output_dir = Path('data/processed/merged')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the merger
    merger = DefenseDataMerger()
    
    # Merge the data
    merger.merge_defense_data()
    
    print("Defense data merge complete!")

if __name__ == "__main__":
    main() 