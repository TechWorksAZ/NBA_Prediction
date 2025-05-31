"""
Column Mapping Module

This module centralizes all column mappings from FINAL_col_mapping.json
to ensure consistency across feature engineering scripts.

The mappings are organized by data type:
1. Player Game Data (pg_data)
2. Player Game Matchup (pg_matchup)
3. Player Game Shotchart (pg_shotchart)
4. Player Game Rest (pg_rest)
5. Player Game Clutch (pg_clutch)
6. Player Game Advanced (pg_advanced)
7. Player Game Defense (pg_defense)
8. Team Game Data (tg_data)
9. Team Game Bench (tg_bench)
10. Team Game Starters (tg_starters)
11. Team Season Data (tt_data)
12. Team Season Clutch (tt_clutch)
13. Validated Games (validated_games)
14. Validated Odds (validated_odds)
15. Team Game Features (team_game)
16. Advanced Stats Features (advanced)
17. Team Specific Features (team_specific)
18. Shot Chart Features (shotchart)
19. Betting Features (betting)
20. Target Features (target)

Author: NBA Prediction Team
Date: 2024
"""

import json
from pathlib import Path
from typing import Dict

# Load column mappings from FINAL_col_mapping.json
def load_column_mappings() -> Dict:
    """Load column mappings from the JSON file."""
    mapping_path = Path("data/processed/columns/FINAL_col_mapping.json")
    with open(mapping_path, 'r') as f:
        mappings = json.load(f)
    return mappings

# Get column mappings for specific data type
def get_columns(data_type: str) -> Dict[str, str]:
    """Get column mappings for a specific data type."""
    mappings = load_column_mappings()
    if data_type in mappings.get('processed_data', {}):
        return {col: col for col in mappings['processed_data'][data_type]['columns']}
    return {}

# Player Game Data Columns
PG_COLS = get_columns('pg_data')

# Player Game Matchup Columns
PG_MATCHUP_COLS = get_columns('pg_matchup')

# Player Game Shotchart Columns
PG_SHOTCHART_COLS = get_columns('pg_shotchart')

# Player Game Rest Columns
PG_REST_COLS = get_columns('pg_rest')

# Player Game Clutch Columns
PG_CLUTCH_COLS = get_columns('pg_clutch')

# Player Game Advanced Columns
PG_ADVANCED_COLS = get_columns('pg_advanced')

# Player Game Defense Columns
PG_DEFENSE_COLS = get_columns('pg_defense')

# Team Game Data Columns
TG_COLS = get_columns('tg_data')

# Team Game Bench Columns
TG_BENCH_COLS = get_columns('tg_bench')

# Team Game Starters Columns
TG_STARTER_COLS = get_columns('tg_starters')

# Team Season Data Columns
TT_COLS = get_columns('tt_data')

# Team Season Clutch Columns
TT_CLUTCH_COLS = get_columns('tt_clutch')

# Validated Games Columns
VALIDATED_GAMES_COLS = get_columns('validated_games')

# Validated Odds Columns
VALIDATED_ODDS_COLS = get_columns('validated_odds')

# Feature Files Columns
TEAM_GAME_COLS = get_columns('team_game')
ADVANCED_COLS = get_columns('advanced')
TEAM_SPECIFIC_COLS = get_columns('team_specific')
SHOTCHART_COLS = get_columns('shotchart')
BETTING_COLS = get_columns('betting')
TARGET_COLS = get_columns('target')

# Helper function to get column or fill with 0 if missing
def get_col(df, col: str) -> str:
    """Get column value or 0 if missing."""
    return df[col] if col in df.columns else 0 