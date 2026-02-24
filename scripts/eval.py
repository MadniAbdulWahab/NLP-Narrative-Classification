import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import evaluate_files

if __name__ == "__main__":
    gold = str(ROOT / "data" / "annotations" / "annotation.txt")
    pred = str(ROOT / "outputs" / "submission.txt")
    evaluate_files(gold, pred)