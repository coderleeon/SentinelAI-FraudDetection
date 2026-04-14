"""Train all models — convenience script for project root execution."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train import run_training_pipeline
if __name__ == "__main__":
    run_training_pipeline()
