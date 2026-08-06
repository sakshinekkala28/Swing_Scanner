"""
Run Swing Scanner without opening the Streamlit UI.

Used by

- GitHub Actions
- Cron Jobs
- Local CLI
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------------------
# Enable headless execution
# -------------------------------------------------------------
os.environ["SWING_SCANNER_HEADLESS"] = "1"

from app.swing_scanner_app import main


def run():

    print("=" * 70)
    print("Swing Scanner")
    print("Headless Mode")
    print("=" * 70)

    main()

    print("\nScan completed.")


if __name__ == "__main__":

    run()