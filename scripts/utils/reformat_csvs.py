import os
import pandas as pd
import csv
from pathlib import Path

# Constants
BASE_DIR = "G:/My Drive/Projects/NBA_Prediction"
DAILY_DIR = os.path.join(BASE_DIR, "data/raw/betting/sbr_daily")

def reformat_csv(file_path: str) -> None:
    """Reformat a single CSV file with consistent formatting."""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Save with consistent formatting
        df.to_csv(
            file_path,
            index=False,
            quoting=csv.QUOTE_ALL,
            line_terminator='\n',
            encoding='utf-8'
        )
        print(f"✅ Reformatted {os.path.basename(file_path)}")
    except Exception as e:
        print(f"⚠️ Error reformatting {os.path.basename(file_path)}: {str(e)}")

def main():
    print("Starting CSV reformatting...")
    
    # Get all CSV files in the daily directory
    csv_files = list(Path(DAILY_DIR).glob("sbr_odds_*.csv"))
    total_files = len(csv_files)
    
    print(f"Found {total_files} CSV files to reformat")
    
    # Reformat each file
    for i, file_path in enumerate(csv_files, 1):
        print(f"\nProcessing file {i}/{total_files}")
        reformat_csv(str(file_path))
    
    print("\n✅ CSV reformatting complete!")

if __name__ == "__main__":
    main() 