import os
import pandas as pd
from datetime import timedelta
from pathlib import Path

# Constants
DATA_BASE_DIR = "C:/Projects/NBA_Prediction"
DATA_DIR = os.path.join(DATA_BASE_DIR, "data/raw/betting")
MASTER_FILE = os.path.join(DATA_DIR, "nba_sbr_odds_2025.csv")

def fix_master_dates():
    """Fix dates in the master file by shifting them one day forward."""
    print(f"📖 Reading master file: {MASTER_FILE}")
    
    # Read the master file
    df = pd.read_csv(MASTER_FILE)
    original_len = len(df)
    
    # Convert date column to datetime, shift forward one day, and back to string
    print("📅 Correcting dates...")
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'] + timedelta(days=1)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Create backup of original file
    backup_file = MASTER_FILE + '.bak'
    print(f"💾 Creating backup at: {backup_file}")
    os.replace(MASTER_FILE, backup_file)
    
    # Save corrected file
    print(f"💾 Saving corrected master file...")
    df.to_csv(MASTER_FILE, index=False, quoting=1, lineterminator='\n', encoding='utf-8')
    
    print(f"✅ Successfully updated {original_len} rows with corrected dates")
    print(f"Original file backed up to: {backup_file}")

if __name__ == "__main__":
    fix_master_dates() 