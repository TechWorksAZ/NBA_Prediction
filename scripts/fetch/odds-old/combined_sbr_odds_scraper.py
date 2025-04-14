import os
import json
import time
import requests
import pandas as pd
import re
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# === CONFIG ===
BASE_DIR = "c:/projects/nba_prediction"
DATA_DIR = os.path.join(BASE_DIR, "data/raw/betting")
DAILY_DIR = os.path.join(DATA_DIR, "sbr_daily")
MASTER_FILE = os.path.join(DATA_DIR, "nba_sbr_odds_2025.csv")
REQUEST_DELAY = 1.5
MAX_RETRIES = 3
START_DATE = datetime(2024, 10, 4)
END_DATE = datetime(2025, 4, 5)

ODDS_TYPES = {
    "point-spread": "spreads",
    "money-line": "moneylines",
    "totals": "totals"
}
GAME_PERIODS = [
    "full-game", "1st-half", "2nd-half",
    "1st-quarter", "2nd-quarter", "3rd-quarter", "4th-quarter"
]

os.makedirs(DAILY_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def extract_odds_data(odds_view, odds_type):
    if not odds_view:
        return {}
    data = {}
    line_data = odds_view.get('currentLine', {})
    if odds_type == 'spreads':
        data.update({
            'home_spread': line_data.get('homeSpread'),
            'away_spread': line_data.get('awaySpread'),
            'home_spread_odds': line_data.get('homeOdds'),
            'away_spread_odds': line_data.get('awayOdds')
        })
    elif odds_type == 'moneylines':
        data.update({
            'home_moneyline': line_data.get('homeOdds'),
            'away_moneyline': line_data.get('awayOdds')
        })
    elif odds_type == 'totals':
        data.update({
            'total': line_data.get('total'),
            'over_odds': line_data.get('overOdds'),
            'under_odds': line_data.get('underOdds')
        })
    opening = odds_view.get('openingLine', {})
    if opening:
        prefix = 'opening_'
        if odds_type == 'spreads':
            data.update({
                f'{prefix}home_spread': opening.get('homeSpread'),
                f'{prefix}away_spread': opening.get('awaySpread'),
                f'{prefix}home_spread_odds': opening.get('homeOdds'),
                f'{prefix}away_spread_odds': opening.get('awayOdds')
            })
        elif odds_type == 'moneylines':
            data.update({
                f'{prefix}home_moneyline': opening.get('homeOdds'),
                f'{prefix}away_moneyline': opening.get('awayOdds')
            })
        elif odds_type == 'totals':
            data.update({
                f'{prefix}total': opening.get('total'),
                f'{prefix}over_odds': opening.get('overOdds'),
                f'{prefix}under_odds': opening.get('underOdds')
            })
    return data

def get_sbr_data_for_date(date_str):
    try:
        base_url = f"https://www.sportsbookreview.com/betting-odds/nba-basketball/?date={date_str}"
        r = requests.get(base_url)
        j = re.findall('__NEXT_DATA__" type="application/json">(.*?)</script>', r.text)
        build_id = json.loads(j[0])['buildId']
        def get_data(odds_type, period):
            url = f"https://www.sportsbookreview.com/_next/data/{build_id}/betting-odds/nba-basketball/{odds_type}/{period}.json?league=nba-basketball&oddsType={odds_type}&oddsScope={period}&date={date_str}"
            return requests.get(url).json()
        result = {"date": date_str}
        for odds_type, odds_key in ODDS_TYPES.items():
            result[odds_key] = {}
            for period in GAME_PERIODS:
                try:
                    data = get_data(odds_type, period)
                    result[odds_key][period] = data
                    print(f"✅ Fetched {odds_type} for {period} on {date_str}")
                    time.sleep(REQUEST_DELAY / 2)
                except Exception as e:
                    print(f"⚠️ Failed to fetch {odds_type} for {period} on {date_str}: {e}")
                    result[odds_key][period] = None
        return result
    except Exception as e:
        print(f"⚠️ Failed on {date_str}: {e}")
        return None

def check_missing_data(data):
    if not data:
        return True
    for odds_key in ODDS_TYPES.values():
        for period in GAME_PERIODS:
            if not data.get(odds_key, {}).get(period):
                return True
    return False

def main():
    current_date = START_DATE
    while current_date <= END_DATE:
        date_str = current_date.strftime("%Y-%m-%d")
        file_path = os.path.join(DAILY_DIR, f"{date_str}.json")
        existing_data = None
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    existing_data = json.load(f)
                except:
                    print(f"⚠️ Failed to load existing {date_str}, re-fetching...")
        if not existing_data or check_missing_data(existing_data):
            print(f"🔄 Fetching odds for {date_str}...")
            data = get_sbr_data_for_date(date_str)
            if data:
                with open(file_path, "w") as f:
                    json.dump(data, f)
                print(f"✅ Saved {date_str}")
        else:
            print(f"✅ {date_str} already complete.")
        time.sleep(REQUEST_DELAY)
        current_date += timedelta(days=1)

    # Merge into master file
    all_files = [f for f in os.listdir(DAILY_DIR) if f.endswith(".json")]
    all_records = []
    for filename in sorted(all_files):
        with open(os.path.join(DAILY_DIR, filename), "r") as f:
            data = json.load(f)
            date = data["date"]
            for odds_type, odds_key in ODDS_TYPES.items():
                odds_data = data.get(odds_key, {})
                if not odds_data:
                    continue
                for period in GAME_PERIODS:
                    period_data = odds_data.get(period)
                    if not period_data:
                        continue
                    try:
                        games = period_data['pageProps']['oddsTables'][0]['oddsTableModel']['gameRows']
                        for game in games:
                            row = {
                                "date": date,
                                "game_id": game['gameView']['gameId'],
                                "odds_type": odds_type,
                                "game_period": period,
                                "home_team": game['gameView']['homeTeam']['fullName'],
                                "away_team": game['gameView']['awayTeam']['fullName'],
                                "home_score": game['gameView'].get('homeTeamScore'),
                                "away_score": game['gameView'].get('awayTeamScore'),
                                "game_status": game['gameView']['gameStatusText'],
                                "start_time": game['gameView']['startDate']
                            }
                            if game['oddsViews']:
                                odds_views = [ov for ov in game['oddsViews'] if ov]  # filter nulls
                                if not odds_views:
                                    continue
                                odds_view = odds_views[0]
                                row['sportsbook'] = odds_view['sportsbook']
                                odds_values = extract_odds_data(odds_view, odds_key)
                                row.update(odds_values)
                            all_records.append(row)
                    except Exception as e:
                        print(f"⚠️ Error processing {filename} for {odds_type} {period}: {e}")
    df = pd.DataFrame(all_records)
    df.to_csv(MASTER_FILE, index=False)
    print(f"✅ Merged all records to {MASTER_FILE}")

if __name__ == "__main__":
    main()
