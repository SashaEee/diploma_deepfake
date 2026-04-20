"""Скачивание обученных весов с Kaggle или по прямой URL.

Использование:
    # Скачать output последнего запуска ноутбука с Kaggle:
    python scripts/download_weights.py --source kaggle --kernel sashaeeeee/edn-ad-training

    # Скачать по прямой URL:
    python scripts/download_weights.py --source url --url https://example.com/best_model.pt
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def download_from_kaggle_kernel(kernel_slug: str, output_dir: str = "checkpoints") -> Path:
    """Скачивает output-файлы Kaggle kernel (ноутбука)."""
    import subprocess

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Скачиваю output ноутбука {kernel_slug} с Kaggle...")
    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_slug, "-p", str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Ошибка Kaggle CLI:\n{result.stderr}", file=sys.stderr)
        print("Убедитесь, что ~/.kaggle/kaggle.json настроен корректно", file=sys.stderr)
        sys.exit(1)

    # Распаковываем ZIP если нужно
    for zf in out.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(out)
        zf.unlink()

    weights = list(out.glob("best_model.pt"))
    if not weights:
        print("best_model.pt не найден в скачанных файлах", file=sys.stderr)
        print(f"Содержимое {out}:", file=sys.stderr)
        for f in out.iterdir():
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)

    path = weights[0]
    print(f"Веса скачаны: {path} ({path.stat().st_size / 1e6:.1f} МБ)")
    return path


def download_from_url(url: str, output_dir: str = "checkpoints") -> Path:
    """Скачивает файл весов по прямой URL."""
    import urllib.request

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "best_model.pt"
    print(f"Скачиваю {url} → {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"Готово: {dest.stat().st_size / 1e6:.1f} МБ")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Скачивание весов EDN-Ad")
    parser.add_argument("--source", choices=["kaggle", "url"], required=True,
                        help="Источник: 'kaggle' — из ноутбука, 'url' — прямая ссылка")
    parser.add_argument("--kernel", default="sashaeeeee/edn-ad-training",
                        help="Kaggle kernel slug (user/kernel-name)")
    parser.add_argument("--url", help="Прямой URL к файлу best_model.pt")
    parser.add_argument("--output-dir", default="checkpoints",
                        help="Директория для сохранения весов (default: checkpoints/)")
    args = parser.parse_args()

    if args.source == "kaggle":
        path = download_from_kaggle_kernel(args.kernel, args.output_dir)
    else:  # url
        if not args.url:
            parser.error("--url обязателен при --source url")
        path = download_from_url(args.url, args.output_dir)

    print(f"\nВеса сохранены: {path}")
    print("\nЗапуск инференса:")
    print("  deepfake-cli single --config configs/base.yaml --input video.mp4")


if __name__ == "__main__":
    main()
