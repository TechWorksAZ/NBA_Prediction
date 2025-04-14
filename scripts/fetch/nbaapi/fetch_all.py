import subprocess
import time
from datetime import datetime
import os
import sys

# === Resolve Full Path to Virtual Environment's Python ===
VENV_PYTHON = os.path.join("env", "Scripts", "python.exe")

# === List of Fetch Scripts to Run ===
FETCH_SCRIPTS = [
    "fetch_player_box_scores.py",
    "fetch_team_box_scores.py",
    "fetch_schedule_data.py",
    "fetch_play_by_play.py",
    "fetch_shot_chart_data.py"
]

def run_script(script_name):
    print(f"\n🚀 Starting {script_name} at {datetime.now().strftime('%H:%M:%S')}")

    try:
        subprocess.run([VENV_PYTHON, f"scripts/fetch/{script_name}"], check=True)
        print(f"✅ Finished {script_name} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with return code: {e.returncode}")
    except Exception as e:
        print(f"🔥 Unexpected error in {script_name}: {e}")

    print("⏳ Waiting 3 seconds before next script...")
    time.sleep(3)

def main():
    print("\n📦 MASTER FETCH RUNNER STARTED")
    start_time = datetime.now()

    for script in FETCH_SCRIPTS:
        run_script(script)

    end_time = datetime.now()
    print("\n✅ ALL FETCHES COMPLETE")
    print(f"🕒 Total runtime: {end_time - start_time}")

if __name__ == "__main__":
    main()
