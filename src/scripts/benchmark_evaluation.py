"""
Legacy Benchmark Evaluation Adapter.
Delegates directly to src.scripts.evaluate_v2.run_isolated_evaluation()
to maintain single source of truth for all out-of-domain benchmarks.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.scripts.evaluate_v2 import run_isolated_evaluation

def run_benchmark():
    run_isolated_evaluation()

if __name__ == "__main__":
    run_benchmark()

