#!/usr/bin/env python3
"""CLI entrypoint: python3 datapipeline/run.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import run

if __name__ == "__main__":
    run()
