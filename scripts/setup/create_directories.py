import os
from pathlib import Path

# Define base directories
BASE_DIR = Path("C:/Projects/NBA_Prediction")
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

# Define subdirectories
SUBDIRS = {
    DATA_DIR: ["raw", "processed"],
    RAW_DIR: [
        "advanced",
        "betting",
        "core",
        "defense",
        "matchups",
        "tracking"
    ],
    RAW_DIR / "betting": ["sbr_daily"],
    PROCESSED_DIR: ["features", "models", "predictions"]
}

def create_directories():
    """Create the project directory structure."""
    print("🚀 Setting up project directories...")
    
    for base_dir, subdirs in SUBDIRS.items():
        for subdir in subdirs:
            dir_path = base_dir / subdir
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created directory: {dir_path}")
            except Exception as e:
                print(f"✗ Error creating {dir_path}: {e}")

if __name__ == "__main__":
    create_directories() 