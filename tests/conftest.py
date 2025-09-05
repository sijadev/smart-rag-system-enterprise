# Ensure project root is on sys.path so tests and modules can import absolute packages like `src`.
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    # Insert at front so it has priority during test collection
    sys.path.insert(0, PROJECT_ROOT)
