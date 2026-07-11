"""CLI: predict tomorrow's market direction from an Excel OHLCV file.

Usage:
    python main.py --input data/ShKol.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_market_data
from src.features import build_training_frame
from src.model import evaluate_model, train_model
from src.predict import predict_next_day

DEFAULT_INPUT = Path(__file__).parent / "data" / "ShKol.xlsx"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict next-day market direction from OHLCV Excel data."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the Excel (.xlsx) file with OHLCV columns.",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Sheet name or index to read (default: first sheet).",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of time-series cross-validation folds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    df = load_market_data(args.input, sheet_name=args.sheet)
    print(f"Loaded {len(df)} trading sessions from {args.input}")

    frame = build_training_frame(df)
    print(f"Built training frame with {len(frame)} labelled samples")

    result = evaluate_model(frame, n_splits=args.splits)
    print("\n=== Time-series cross-validation ===")
    print(f"Folds:              {result.n_splits}")
    print(f"Accuracy:           {result.accuracy:.4f}")
    print(f"F1 (up class):      {result.f1:.4f}")
    print(f"Baseline accuracy:  {result.baseline_accuracy:.4f}")

    model = train_model(frame)
    prediction = predict_next_day(model, df)

    print("\n=== Next-day prediction ===")
    print(f"Last close:         {prediction.last_close:,.1f}")
    print(f"Direction:          {prediction.direction} ({prediction.label_fa})")
    print(f"P(up):              {prediction.probability_up:.4f}")


if __name__ == "__main__":
    main()
