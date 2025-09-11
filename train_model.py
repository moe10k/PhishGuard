#!/usr/bin/env python3
"""Training script for PhishGuard model."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training import main

if __name__ == "__main__":
    main()
