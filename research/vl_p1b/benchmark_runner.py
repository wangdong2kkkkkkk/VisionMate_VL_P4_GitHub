"""Root wrapper for running VL-P1b benchmark & ablation suite."""
from src.benchmark_runner import run_all_benchmarks
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    run_all_benchmarks(root / "research" / "vl_p1", root / "research" / "vl_p1b")
