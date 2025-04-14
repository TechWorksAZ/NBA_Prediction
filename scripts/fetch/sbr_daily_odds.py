import os
import json
import time
import pandas as pd
import csv
from datetime import datetime, timedelta
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from typing import Dict, List, Optional

# Constants
BASE_DIR = "G:/My Drive/Projects/NBA_Prediction"  # Updated to match your local path
DATA_DIR = os.path.join("c:/projects/nba_prediction/data/raw/betting")
DAILY_DIR = os.path.join("c:/projects/nba_prediction/data/raw/betting/sbr_daily")
SPORT = "NBA"
REQUEST_DELAY = 1.0
MAX_RETRIES = 3
RETRY_DELAY = 3
PAGE_LOAD_TIMEOUT = 20

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

ODDS_TYPES = {
    'spreads': 'pointspread',
    'moneylines': 'money-line',
    'totals': 'totals'
}

# Ensure directories exist
os.makedirs(DAILY_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

class SBRScraper:
    def __init__(self):
        self.driver = None
        
    def setup_driver(self):
        """Set up the undetected Chrome WebDriver with robust error handling."""
        if self.driver is not None:
            return self.driver
            
        try:
            print("🔧 Setting up Chrome driver...")
            
            # Configure Chrome options
            options = uc.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--disable-blink-features=AutomationControlled')
            
            # Add headers to appear more like a real browser
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36')
            
            print("🌐 Initializing Chrome driver...")
            self.driver = uc.Chrome(
                options=options,
                driver_executable_path=None,
                browser_executable_path=None,
                version_main=135
            )
            
            # Set page load timeout
            self.driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
            
            # Add a wait condition for JavaScript to load
            self.driver.implicitly_wait(10)
            
            print("✅ Chrome driver setup complete")
            return self.driver
            
        except Exception as e:
            print(f"⚠️ Error setting up Chrome driver: {str(e)}")
            print("\nTroubleshooting tips:")
            print("1. Make sure Google Chrome is installed")
            print("2. Try running without --headless if issues persist")
            print("3. Check if Chrome browser needs updating")
            return None

    def cleanup(self):
        """Clean up the WebDriver instance."""
        if self.driver is not None:
            try:
                self.driver.quit()
            except:
                pass
            finally:
                self.driver = None

    def get_sbr_data_for_date(self, date_str: str) -> Optional[Dict]:
        """Fetch SBR odds data for a specific date using Selenium."""
        # Check if data already exists for this date
        daily_file = os.path.join(DAILY_DIR, f"sbr_odds_{date_str}.csv")
        if os.path.exists(daily_file):
            print(f"✅ Data already exists for {date_str}, skipping...")
            return None
        
        if not self.setup_driver():
            print("⚠️ Failed to initialize Chrome driver")
            return None
            
            data = {
                'date': date_str,
            'games': {},  # Store game metadata here
                'spreads': {},
                'moneylines': {},
                'totals': {}
            }
        
        try:
            # Base URL for NBA odds
            base_url = "https://www.sportsbookreview.com/betting-odds/nba-basketball/"
        
        # Fetch data for each period and odds type
        for period in GAME_PERIODS:
                for odds_type, odds_path in ODDS_TYPES.items():
                    print(f"📥 Fetching {period} - {odds_type} for {date_str}...")
                    
                    # Construct URL based on period and odds type
                    if period == 'full-game' and odds_type == 'spreads':
                    url = f"{base_url}?date={date_str}"
                else:
                    url = f"{base_url}{odds_path}/{period}/?date={date_str}"

                    # Load the page with retries
                    for attempt in range(MAX_RETRIES):
                        try:
                            print(f"🌐 Loading page for {period} - {odds_type} (attempt {attempt + 1})...")
                            self.driver.get(url)
                            
                            # Find the script tag containing the JSON data
                            script_element = self.driver.find_element(By.ID, "__NEXT_DATA__")
                            json_text = script_element.get_attribute("textContent")
                            json_data = json.loads(json_text)
                            
                            # Extract games data with error handling
                            try:
                                games = json_data['props']['pageProps']['oddsTables'][0]['oddsTableModel']['gameRows']
                                print(f"Found {len(games)} games in the data")
                            except (KeyError, IndexError) as e:
                                print(f"Error accessing games data: {str(e)}")
                                continue
                            
                            for game in games:
                                try:
                                    game_view = game['gameView']
                                    game_id = game_view['gameId']
                                    
                                    # Store game metadata if not already stored
                                    if game_id not in data['games']:
                                        data['games'][game_id] = game_view
                                    
                                    # Extract odds data
                                    if 'oddsViews' in game:
                                        for odds_view in game['oddsViews']:
                                            if not odds_view:  # Skip null odds views
                                                continue
                                                
                                            sportsbook = odds_view.get('sportsbook')
                                            if not sportsbook:  # Skip if no sportsbook
                        continue

                                            current_line = odds_view.get('currentLine', {})
                                            if not current_line:  # Skip if no current line
                                                continue

                                            # Initialize data structures if needed
                                            if odds_type not in data:
                                                data[odds_type] = {}
                                            if period not in data[odds_type]:
                                                data[odds_type][period] = {}
                                            if sportsbook not in data[odds_type][period]:
                                                data[odds_type][period][sportsbook] = {}
                                            
                                            # Spread odds
                                            if odds_type == 'spreads':
                                                if current_line.get('homeSpread') is not None:
                                                    data[odds_type][period][sportsbook][game_id] = {
                                                        'home_spread': current_line['homeSpread'],
                                                        'away_spread': current_line['awaySpread'],
                                                        'home_odds': current_line['homeOdds'],
                                                        'away_odds': current_line['awayOdds']
                                                    }
                                            
                                            # Moneyline odds
                                            elif odds_type == 'moneylines':
                                                if current_line.get('homeOdds') is not None:
                                                    data[odds_type][period][sportsbook][game_id] = {
                                                        'home_odds': current_line['homeOdds'],
                                                        'away_odds': current_line['awayOdds']
                                                    }
                                            
                                            # Total odds
                                            elif odds_type == 'totals':
                                                if current_line.get('total') is not None:
                                                    data[odds_type][period][sportsbook][game_id] = {
                                                        'total': current_line['total'],
                                                        'over_odds': current_line['overOdds'],
                                                        'under_odds': current_line['underOdds']
                                                    }
                                except Exception as e:
                                    print(f"Error processing game: {str(e)}")
                    continue

                            print(f"✅ Successfully extracted data for {len(games)} games in {period} - {odds_type}")
                            break
                            
                        except Exception as e:
                            print(f"Attempt {attempt + 1} failed: {str(e)}")
                            if attempt < MAX_RETRIES - 1:
                                time.sleep(RETRY_DELAY)
                    continue

        return data
        
    except Exception as e:
        print(f"⚠️ Error fetching data for {date_str}: {str(e)}")
        return None

def process_game_data(data: Dict, date_str: str) -> List[Dict]:
    """Process the game data and extract relevant information."""
    games = []
    
    # Get all unique game IDs from the odds data
    game_ids = set()
    for odds_type in data:
        if odds_type in ['date', 'games']:
            continue
        for period in data[odds_type]:
            for sportsbook in data[odds_type][period]:
                game_ids.update(data[odds_type][period][sportsbook].keys())
    
    for game_id in game_ids:
        # Get game info from stored metadata
        game_info = data['games'].get(game_id)
        if not game_info:
            print(f"⚠️ No game info found for game ID: {game_id}")
            continue
            
        try:
            game_data = {
                'game_id': game_id,
                'date': date_str,
                'away_team': game_info['awayTeam']['name'],
                'home_team': game_info['homeTeam']['name'],
                'start_time': game_info['startDate'],
                'venue': game_info['venueName'],
                'city': game_info['city'],
                'state': game_info['state'],
                'status': game_info['status'],
                'away_score': game_info['awayTeamScore'],
                'home_score': game_info['homeTeamScore']
            }
            
            # Add odds data for each period and odds type
            for odds_type in ['spreads', 'moneylines', 'totals']:
                for period in GAME_PERIODS:
                    for sportsbook in data.get(odds_type, {}).get(period, {}):
                        if game_id in data[odds_type][period][sportsbook]:
                            odds = data[odds_type][period][sportsbook][game_id]
                            if odds_type == 'spreads':
                                game_data.update({
                                    f'{period}_{sportsbook}_spread_away': odds['away_spread'],
                                    f'{period}_{sportsbook}_spread_away_odds': odds['away_odds'],
                                    f'{period}_{sportsbook}_spread_home': odds['home_spread'],
                                    f'{period}_{sportsbook}_spread_home_odds': odds['home_odds']
                                })
                            elif odds_type == 'moneylines':
                                game_data.update({
                                    f'{period}_{sportsbook}_moneyline_away': odds['away_odds'],
                                    f'{period}_{sportsbook}_moneyline_home': odds['home_odds']
                                })
                            elif odds_type == 'totals':
                                game_data.update({
                                    f'{period}_{sportsbook}_total': odds['total'],
                                    f'{period}_{sportsbook}_total_over_odds': odds['over_odds'],
                                    f'{period}_{sportsbook}_total_under_odds': odds['under_odds']
                                })
            
            games.append(game_data)
            
        except Exception as e:
            print(f"⚠️ Error processing game {game_id}: {str(e)}")
            continue
    
    return games

def save_daily_data(data: Dict, date_str: str) -> None:
    """Save the daily data to a file."""
    if not data:
        return
        
    # Process the game data
    games = process_game_data(data, date_str)
    
    if not games:
        print("⚠️ No games to save")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(games)
    
    # Save to CSV with consistent formatting
    filename = os.path.join(DAILY_DIR, f"sbr_odds_{date_str}.csv")
    df.to_csv(
        filename, 
        index=False,
        quoting=csv.QUOTE_ALL,
        lineterminator='\n',
        encoding='utf-8'
    )
    print(f"✅ Saved daily data to {filename}")

def get_date_range(start_date: str, end_date: str) -> List[str]:
    """Generate a list of dates between start_date and end_date (inclusive)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = []
    
    current = start
    while current <= end:
        date_list.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return date_list

def main():
    # Set date range
    start_date = "2025-04-13"
    end_date = "2025-04-13"  # Yesterday's date
    date_list = get_date_range(start_date, end_date)
    
    print(f"\nFetching data from {start_date} to {end_date}")
    print(f"Total dates to process: {len(date_list)}")
    print(f"Data will be saved to: {DAILY_DIR}")
    
    scraper = SBRScraper()
    
    try:
        for date_str in date_list:
            print(f"\nProcessing date: {date_str}")
            
            # Get data for target date
            data = scraper.get_sbr_data_for_date(date_str)
            
            if not data:
                print(f"⚠️ No data fetched for {date_str}")
                continue
            
            # Save the data
            save_daily_data(data, date_str)
            print(f"✅ Data saved for {date_str}")
            
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    main()
