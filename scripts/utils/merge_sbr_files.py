import os
import pandas as pd
import csv
from pathlib import Path
from datetime import datetime, timedelta

# Constants
BASE_DIR = "G:/My Drive/Projects/NBA_Prediction"
DATA_BASE_DIR = "C:/Projects/NBA_Prediction"
DATA_DIR = os.path.join(DATA_BASE_DIR, "data/raw/betting")
DAILY_DIR = os.path.join(DATA_DIR, "sbr_daily")
MASTER_FILE = os.path.join(DATA_DIR, "nba_sbr_odds_2025.csv")

def correct_game_date(df: pd.DataFrame) -> pd.DataFrame:
    """Correct game dates by shifting them one day forward."""
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'] + timedelta(days=1)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df

def merge_all_files():
    """Merge all daily SBR files into a master file."""
    print("🔍 Finding all daily SBR files...")
    
    # Get all CSV files in the daily directory
    daily_files = list(Path(DAILY_DIR).glob("sbr_odds_*.csv"))
    if not daily_files:
        print("⚠️ No daily files found!")
        return
        
    print(f"📂 Found {len(daily_files)} daily files")
    
    # Read and concatenate all files
    all_data = []
    for file_path in daily_files:
        try:
            df = pd.read_csv(file_path)
            # Correct the game date
            df = correct_game_date(df)
            all_data.append(df)
            print(f"✅ Read and corrected dates for {file_path.name}")
        except Exception as e:
            print(f"⚠️ Error reading {file_path.name}: {str(e)}")
            continue
    
    if not all_data:
        print("⚠️ No data to merge!")
        return
    
    # Concatenate all dataframes
    print("🔄 Merging all files...")
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates based on game_id and date
    master_df = master_df.drop_duplicates(subset=['game_id', 'date'])
    
    # Sort by date and game_id
    master_df = master_df.sort_values(['date', 'game_id'])
    
    # Save to master file
    print(f"💾 Saving to master file: {MASTER_FILE}")
    master_df.to_csv(
        MASTER_FILE,
        index=False,
        quoting=csv.QUOTE_ALL,
        lineterminator='\n',
        encoding='utf-8'
    )
    
    print(f"✅ Successfully merged {len(daily_files)} files into master file")
    print(f"📊 Total rows in master file: {len(master_df)}")

if __name__ == "__main__":
    merge_all_files() 