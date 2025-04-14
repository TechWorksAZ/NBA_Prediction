from bs4 import BeautifulSoup

def parse_games_from_html(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        print("\nAnalyzing HTML structure...\n")
        
        # Find all game containers
        game_containers = soup.find_all('div', class_='event-card')
        print(f"Found {len(game_containers)} game containers\n")
        
        games = []
        for container in game_containers:
            try:
                print(f"Processing game {container.get('data-event-id', 'unknown')}")
                
                # Extract team names
                team_names = []
                team_divs = container.find_all('div', class_='event-card__participant-name')
                for team_div in team_divs:
                    team_name = team_div.get_text(strip=True)
                    print(f"Found team: {team_name}")
                    team_names.append(team_name)
                print(f"Found teams: {team_names}")
                
                if len(team_names) != 2:
                    print("❌ Error: Did not find exactly 2 teams")
                    continue
                
                # Initialize game data
                game_data = {
                    'game_id': container.get('data-event-id', ''),
                    'away_team': team_names[0],
                    'home_team': team_names[1],
                    'odds': {
                        'away_odds': None,
                        'home_odds': None,
                        'away_spread': None,
                        'home_spread': None,
                        'total_over': None,
                        'total_under': None
                    }
                }
                
                # Find the markets container
                markets_container = container.find('div', class_='event-card__markets')
                if markets_container:
                    print("Found markets container")
                    
                    # Process each market type
                    market_types = markets_container.find_all('div', class_='event-card__market')
                    for market in market_types:
                        # Get market name from the header
                        market_header = market.find('div', class_='event-card__market-name')
                        if not market_header:
                            continue
                            
                        market_name = market_header.get_text(strip=True).lower()
                        print(f"Processing market: {market_name}")
                        
                        # Find market options container
                        options_container = market.find('div', class_='event-card__market-options')
                        if not options_container:
                            continue
                            
                        # Get all options
                        options = options_container.find_all('div', class_='event-card__market-option')
                        if len(options) != 2:
                            print(f"❌ Expected 2 options for {market_name}, found {len(options)}")
                            continue
                            
                        for i, option in enumerate(options):
                            try:
                                # Extract line and price
                                line_elem = option.find('div', class_='event-card__option-line')
                                price_elem = option.find('div', class_='event-card__option-price')
                                
                                line = line_elem.get_text(strip=True) if line_elem else None
                                price = price_elem.get_text(strip=True) if price_elem else None
                                
                                print(f"Option {i+1}: Line={line}, Price={price}")
                                
                                # Map values based on market type
                                if 'spread' in market_name:
                                    if i == 0:  # Away team
                                        game_data['odds']['away_spread'] = line
                                        game_data['odds']['away_odds'] = price
                                    else:  # Home team
                                        game_data['odds']['home_spread'] = line
                                        game_data['odds']['home_odds'] = price
                                elif 'total' in market_name:
                                    if i == 0:  # Over
                                        game_data['odds']['total_over'] = line
                                    else:  # Under
                                        game_data['odds']['total_under'] = line
                                elif 'money' in market_name or 'moneyline' in market_name:
                                    if i == 0:  # Away team
                                        game_data['odds']['away_odds'] = price
                                    else:  # Home team
                                        game_data['odds']['home_odds'] = price
                            except Exception as e:
                                print(f"❌ Error processing option {i+1}: {str(e)}")
                                continue
                else:
                    print("❌ No markets container found")
                
                print(f"Processed game: {game_data}")
                games.append(game_data)
                print()  # Add blank line between games
                
            except Exception as e:
                print(f"❌ Error processing game: {str(e)}")
                continue
        
        print(f"✅ Successfully extracted {len(games)} games")
        if len(games) > 0:
            print(f"   ✅ Successfully parsed {len(games)} games")
        return games, html_content
        
    except Exception as e:
        print(f"❌ Error parsing HTML: {str(e)}")
        return [], html_content 