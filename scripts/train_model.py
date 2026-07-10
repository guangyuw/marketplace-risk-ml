#!/usr/bin/env python3
"""CLI entrypoint for training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train_pipeline

if __name__ == "__main__":
    train_pipeline()
