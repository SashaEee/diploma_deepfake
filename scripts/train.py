"""Точка входа для обучения модели."""
import argparse

from deepfake_detector.utils.config import load_config
from deepfake_detector.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--manifest", required=True, help="Path to train manifest JSON")
    parser.add_argument("--val-manifest", help="Path to val manifest JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    from deepfake_detector.models.full_model import DeepfakeDetector
    from deepfake_detector.training.dataset import VideoDataset
    from deepfake_detector.training.trainer import Trainer

    model = DeepfakeDetector(cfg.model)
    train_ds = VideoDataset(args.manifest)
    val_manifest = args.val_manifest or args.manifest
    val_ds = VideoDataset(val_manifest)

    trainer = Trainer(cfg.training, model, train_ds, val_ds)
    trainer.fit()


if __name__ == "__main__":
    main()
