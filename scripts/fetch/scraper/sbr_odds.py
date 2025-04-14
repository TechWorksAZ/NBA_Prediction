# scripts/fetch/sbr_odds.py

from datetime import datetime, timedelta
import requests
import re
import json
import pandas as pd
import os

sport_dict = {
    "NBA": "nba-basketball",
    "NFL": "nfl-football",
    "NHL": "nhl-hockey",
    "MLB": "mlb-baseball",
    "NCAAB": "ncaa-basketball"
}

def fetch_sbr_odds(sport='NBA', date=None, current_line=True):
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")

    _line = 'currentLine' if current_line else 'openingLine'
    odds = []

    try:
        url = f"https://www.sportsbookreview.com/betting-odds/{sport_dict[sport]}/?date={date}"
        r = requests.get(url)
        j = re.findall('__NEXT_DATA__" type="application/json">(.*?)</script>', r.text)
        build_id = json.loads(j[0])['buildId']

        def get_data(endpoint):
            endpoint_url = f"https://www.sportsbookreview.com/_next/data/{build_id}/betting-odds/{sport_dict[sport]}/{endpoint}.json?league={sport_dict[sport]}&oddsType={endpoint}&oddsScope=full-game&date={date}"
            return requests.get(endpoint_url).json()['pageProps']['oddsTables'][0]['oddsTableModel']['gameRows']

        spreads = {g['gameView']['gameId']: g for g in get_data("spreads")}
        moneylines = {g['gameView']['gameId']: g for g in get_data("money-line")}
        totals = {g['gameView']['gameId']: g for g in get_data("totals")}

        all_stats = {
            gid: {
                "spreads": spreads.get(gid),
                "moneylines": moneylines.get(gid),
                "totals": totals.get(gid),
            }
            for gid in spreads
        }

        for gid, event in all_stats.items():
            game = {
                "date": event['spreads']['gameView']['startDate'],
                "home_team": event['spreads']['gameView']['homeTeam']['fullName'],
                "away_team": event['spreads']['gameView']['awayTeam']['fullName'],
                "home_score": event['spreads']['gameView']['homeTeamScore'],
                "away_score": event['spreads']['gameView']['awayTeamScore']
            }

            for line in event.get("spreads", {}).get("oddsViews", []):
                book = line.get("sportsbook")
                if book:
                    game[f"{book}_home_spread"] = line[_line].get("homeSpread")
                    game[f"{book}_away_spread"] = line[_line].get("awaySpread")
                    game[f"{book}_home_spread_odds"] = line[_line].get("homeOdds")
                    game[f"{book}_away_spread_odds"] = line[_line].get("awayOdds")

            for line in event.get("moneylines", {}).get("oddsViews", []):
                book = line.get("sportsbook")
                if book:
                    game[f"{book}_home_ml"] = line[_line].get("homeOdds")
                    game[f"{book}_away_ml"] = line[_line].get("awayOdds")

            for line in event.get("totals", {}).get("oddsViews", []):
                book = line.get("sportsbook")
                if book:
                    game[f"{book}_total"] = line[_line].get("total")
                    game[f"{book}_over_odds"] = line[_line].get("overOdds")
                    game[f"{book}_under_odds"] = line[_line].get("underOdds")

            odds.append(game)

        return pd.DataFrame(odds)

    except Exception as e:
        print(f"⚠️ Error fetching odds for {date}: {e}")
        return pd.DataFrame()
