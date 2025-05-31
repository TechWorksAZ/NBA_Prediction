import pandas as pd
from pathlib import Path, PureWindowsPath
import logging
from typing import Dict, List, Tuple
import shutil
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_relative_path(full_path: str) -> Path:
    """Extract relative path from full Windows path."""
    # Convert Windows path to forward slashes
    normalized_path = full_path.replace('\\', '/')
    
    # Find the 'data/raw' part and extract everything after it
    if 'data/raw' in normalized_path:
        parts = normalized_path.split('data/raw/')
        if len(parts) > 1:
            return Path(parts[1])
    return Path(normalized_path)

def load_column_mappings(mapping_file: Path) -> Dict[str, Dict[str, str]]:
    """Load column mappings from a CSV file."""
    df = pd.read_csv(mapping_file)
    
    # Convert the DataFrame to a dictionary of mappings
    mappings = {}
    for _, row in df.iterrows():
        # Extract relative path from full Windows path
        relative_path = extract_relative_path(row['file_name'])
        
        # Create a mapping from the row, excluding the filename and NaN values
        mapping = {f'col_{i+1}': val for i, val in enumerate(row.values[1:]) 
                  if pd.notna(val)}
        mappings[str(relative_path)] = mapping
    
    return mappings

def normalize_column_names(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Normalize column names according to the mapping."""
    # Create reverse mapping from col_N to actual column name
    reverse_mapping = {val: df.columns[i] for i, val in enumerate(df.columns)}
    
    # Create mapping from current column names to desired names
    name_mapping = {}
    for col_n, target_name in column_mapping.items():
        if col_n in reverse_mapping:
            name_mapping[reverse_mapping[col_n]] = target_name
    
    return df.rename(columns=name_mapping)

def process_file(input_path: Path, output_path: Path, column_mapping: Dict[str, str]) -> None:
    """Process a single file with column normalization."""
    if not input_path.is_file():
        logger.warning(f"File not found: {input_path}")
        return
        
    logger.info(f"Processing file {input_path}")
    try:
        df = pd.read_csv(input_path)
        
        # Apply column mapping if available
        if column_mapping:
            df = normalize_column_names(df, column_mapping)
        
        # Save processed data
        logger.info(f"Saving processed data to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    except Exception as e:
        logger.error(f"Error processing file {input_path}: {str(e)}")

def process_directory(input_dir: Path, output_dir: Path, column_mapping: Dict[str, str]) -> None:
    """Process all files in a directory and its subdirectories."""
    if not input_dir.is_dir():
        logger.warning(f"Directory not found: {input_dir}")
        return
        
    logger.info(f"Processing directory {input_dir}")
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all CSV files in this directory and subdirectories
        for input_file in input_dir.rglob("*.csv"):
            # Maintain the same directory structure in output
            relative_path = input_file.relative_to(input_dir)
            output_file = output_dir / relative_path
            process_file(input_file, output_file, column_mapping)
            
    except Exception as e:
        logger.error(f"Error processing directory {input_dir}: {str(e)}")

def get_data_type_mappings() -> List[Tuple[str, str]]:
    """Get list of (data_type, mapping_file) pairs."""
    return [
        ("core", "core_column_names.csv"),
        ("advanced", "advanced_column_names.csv"),
        ("defense", "defense_column_names.csv"),
        ("tracking", "tracking_column_names.csv"),
        ("matchups", "matchups_betting_column_names.csv"),
        ("betting", "matchups_betting_column_names.csv")
    ]

def main():
    # Process each data type
    for data_type, mapping_file in get_data_type_mappings():
        logger.info(f"Processing {data_type} data")
        
        # Load column mappings
        mapping_path = Path(f"data/processed/columns/{mapping_file}")
        if not mapping_path.exists():
            logger.warning(f"Mapping file not found: {mapping_path}")
            continue
            
        column_mappings = load_column_mappings(mapping_path)
        
        # Process each file/folder in mappings
        for file_path_str, mapping in column_mappings.items():
            file_path = Path(file_path_str)
            raw_path = Path("data/raw") / file_path
            processed_path = Path("data/processed") / file_path
            
            # Check if it's a directory (ends with / or exists as a directory)
            is_dir = str(file_path).endswith('/') or (raw_path.exists() and raw_path.is_dir())
            
            if is_dir:
                # Remove trailing slash if present and process directory
                dir_path = Path(str(file_path).rstrip('/'))
                raw_dir = Path("data/raw") / dir_path
                processed_dir = Path("data/processed") / dir_path
                process_directory(raw_dir, processed_dir, mapping)
            else:
                # Process single file
                process_file(raw_path, processed_path, mapping)
    
    logger.info("Data processing complete")

if __name__ == "__main__":
    main() 