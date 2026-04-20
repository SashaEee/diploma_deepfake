"""Оценка модели на размеченном наборе данных."""
import argparse

from deepfake_detector.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", help="Path to checkpoint .pt file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Evaluating with config: {args.config}")
    print("Not yet fully implemented — will be completed in Stage 6.")


if __name__ == "__main__":
    main()
