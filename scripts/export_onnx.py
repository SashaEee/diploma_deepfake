"""Экспорт модели в ONNX."""
import argparse
import torch

from deepfake_detector.utils.config import load_config
from deepfake_detector.models.full_model import DeepfakeDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="deepfake_detector.onnx")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = DeepfakeDetector(cfg.model)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    dummy = torch.zeros(1, cfg.preprocessing.target_frames, 3, 224, 224)
    torch.onnx.export(
        model, dummy, args.output,
        input_names=["video"], output_names=["logits"],
        dynamic_axes={"video": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
