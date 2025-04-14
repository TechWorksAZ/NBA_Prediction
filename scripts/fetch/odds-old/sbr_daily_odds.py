import os
import json
import time
import requests
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Constants
BASE_DIR = "c:/projects/nba_prediction"
DATA_DIR = os.path.join(BASE_DIR, "data/raw/betting")
DAILY_DIR = os.path.join(DATA_DIR, "sbr_daily")
MASTER_FILE = os.path.join(DATA_DIR, "nba_sbr_odds_2025.csv")
SPORT = "NBA"
REQUEST_DELAY = 1.5  # Delay between requests
MAX_RETRIES = 3  # Maximum number of retries for failed requests
RETRY_DELAY = 5  # Delay between retries in seconds

# Ensure directories exist
os.makedirs(DAILY_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Game periods and odds types
GAME_PERIODS = [
    'full-game',
    '1st-half',
    '2nd-half',
    '1st-quarter',
    '2nd-quarter',
    '3rd-quarter',
    '4th-quarter'
]

ODDS_TYPES = [
    'spreads',
    'moneylines',
    'totals'
]

def create_session():
    """Create a requests session with retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def validate_response_data(data):
    """Validate the structure of the response data."""
    if not isinstance(data, dict):
        return False
        
    # Check for required top-level keys
    required_keys = ['pageProps']
    if not all(key in data for key in required_keys):
        return False
        
    # Check pageProps structure
    page_props = data.get('pageProps', {})
    if not isinstance(page_props, dict):
        return False
        
    # Check for oddsTables
    odds_tables = page_props.get('oddsTables', [])
    if not isinstance(odds_tables, list):
        return False
        
    return True

def get_sbr_data_for_date(date_str, missing_data=None):
    """Fetch SBR odds data for a specific date."""
    base_url = "https://www.sportsbookreview.com/betting-odds/nba-basketball/"
    session = create_session()
    
    try:
        # Initialize or load existing data structure
        if missing_data:
            data = missing_data
        else:
            data = {
                'date': date_str,
                'spreads': {},
                'moneylines': {},
                'totals': {}
            }
        
        # Fetch data for each period and odds type
        for period in GAME_PERIODS:
            for odds_type in ODDS_TYPES:
                if data[odds_type].get(period):
                    print(f"⏩ Skipping {period} - {odds_type} for {date_str} - already exists")
                    continue

                # Determine correct odds path
                if odds_type == 'spreads':
                    odds_path = 'pointspread'
                elif odds_type == 'moneylines':
                    odds_path = 'money-line'
                elif odds_type == 'totals':
                    odds_path = 'totals'
                else:
                    continue

                # Build the URL
                if odds_type == 'spreads' and period == 'full-game':
                    # Special case: full-game spreads are on the base page
                    url = f"{base_url}?date={date_str}"
                else:
                    url = f"{base_url}{odds_path}/{period}/?date={date_str}"

                # Params (if you still want to include region/country info)
                params = {
                    'region': 'ny',
                    'country': 'us'
                }

                try:
                    print(f"📥 Fetching {period} - {odds_type} for {date_str}...")
                    response = session.get(url, params=params)
                    response.raise_for_status()
                    period_data = response.json()

                    if not validate_response_data(period_data):
                        print(f"⚠️ Invalid data structure for {period} - {odds_type} for {date_str}")
                        continue

                    data[odds_type][period] = period_data
                    time.sleep(REQUEST_DELAY)

                except requests.exceptions.RequestException as e:
                    print(f"⚠️ Error fetching {period} - {odds_type} for {date_str}: {str(e)}")
                    continue
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error parsing JSON for {period} - {odds_type} for {date_str}: {str(e)}")
                    continue

        return data
        
    except Exception as e:
        print(f"⚠️ Error fetching data for {date_str}: {str(e)}")
        return None

def load_existing_data(date_str):
    """Load existing data from JSON file if it exists."""
    filename = os.path.join(DAILY_DIR, f"{date_str}.json")
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                # Validate the loaded data
                if validate_response_data(data):
                    return data
                else:
                    print(f"⚠️ Invalid data structure in existing file for {date_str}")
                    return None
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parsing JSON for {date_str}: {str(e)}")
            return None
        except Exception as e:
            print(f"⚠️ Error loading existing data for {date_str}: {str(e)}")
            return None
    return None

def check_missing_data(data):
    """Check which periods and odds types are missing from the data."""
    if not data:
        return True, None  # Need to fetch everything
        
    missing = {
        'spreads': [],
        'moneylines': [],
        'totals': []
    }
    
    has_missing = False
    
    for odds_type in ODDS_TYPES:
        for period in GAME_PERIODS:
            if not data.get(odds_type, {}).get(period):
                missing[odds_type].append(period)
                has_missing = True
                
    return has_missing, missing

def save_daily_data(date_str, data):
    """Save fetched data to a JSON file."""
    filename = os.path.join(DAILY_DIR, f"{date_str}.json")
    
    try:
        # Create a backup of the existing file if it exists
        if os.path.exists(filename):
            backup_filename = f"{filename}.bak"
            os.replace(filename, backup_filename)
            
        # Save the new data
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved data for {date_str} to {filename}")
        
        # Remove the backup if save was successful
        backup_filename = f"{filename}.bak"
        if os.path.exists(backup_filename):
            os.remove(backup_filename)
            
        return True
    except Exception as e:
        print(f"⚠️ Error saving data for {date_str}: {str(e)}")
        # Restore from backup if save failed
        backup_filename = f"{filename}.bak"
        if os.path.exists(backup_filename):
            os.replace(backup_filename, filename)
            print(f"✅ Restored backup for {date_str}")
        return False

def main():
    """Main function to fetch and save odds data."""
    print(f"📁 Saving data to: {DAILY_DIR}")
    print(f"📁 Master file will be: {MASTER_FILE}")
    
    # Date range for the 2024-25 NBA season
    start_date = datetime(2024, 10, 4)
    end_date = datetime(2025, 4, 5)
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Load existing data if any
        existing_data = load_existing_data(date_str)
        has_missing, missing_data = check_missing_data(existing_data)
        
        if not has_missing:
            print(f"✅ {date_str} - All data exists")
            current_date += timedelta(days=1)
            continue
            
        # Fetch missing data
        print(f"\n📥 Fetching missing data for {date_str}...")
        if missing_data:
            for odds_type, periods in missing_data.items():
                if periods:
                    print(f"Missing {odds_type}: {', '.join(periods)}")
                    
        data = get_sbr_data_for_date(date_str, existing_data)
        
        if data:
            # Save data
            if save_daily_data(date_str, data):
                print(f"✅ Successfully updated data for {date_str}")
                
        # Delay between dates
        time.sleep(REQUEST_DELAY)
        current_date += timedelta(days=1)

if __name__ == "__main__":
    main()