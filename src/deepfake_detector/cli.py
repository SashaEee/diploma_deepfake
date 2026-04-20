"""CLI утилита на Typer — четыре команды: single, batch, eval, adapt."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import track

app = typer.Typer(name="deepfake-cli", add_completion=False)
console = Console()


@app.command()
def single(
    config: str = typer.Option("configs/base.yaml", "--config", help="Path to YAML config"),
    input: str = typer.Option(..., "--input", help="Path to video/image file"),
    gradcam: bool = typer.Option(False, "--gradcam", help="Generate Grad-CAM overlay"),
    output: Optional[str] = typer.Option(None, "--output", help="Save JSON result to file"),
):
    """Обработка одного файла."""
    from deepfake_detector.inference.predictor import Predictor

    predictor = Predictor.from_config(config)
    path = Path(input)
    if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        import cv2
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        result = predictor.predict_image(img)
    else:
        result = predictor.predict_video(path)

    console.print_json(json.dumps(result))
    if output:
        Path(output).write_text(json.dumps(result, indent=2))


@app.command()
def batch(
    config: str = typer.Option("configs/base.yaml", "--config"),
    input_dir: str = typer.Option(..., "--input-dir"),
    output: str = typer.Option("report.json", "--output"),
):
    """Пакетная обработка каталога."""
    from deepfake_detector.inference.predictor import Predictor

    predictor = Predictor.from_config(config)
    paths = list(Path(input_dir).rglob("*.mp4")) + list(Path(input_dir).rglob("*.avi"))
    results = []
    for p in track(paths, description="Processing..."):
        res = predictor.predict_video(p)
        res["path"] = str(p)
        results.append(res)
    Path(output).write_text(json.dumps(results, indent=2))
    console.print(f"[green]Done. Results saved to {output}[/green]")


@app.command()
def eval(
    config: str = typer.Option("configs/base.yaml", "--config"),
    manifest: str = typer.Option(..., "--manifest"),
    metrics: str = typer.Option("auroc,f1,eer", "--metrics"),
):
    """Оценка на валидационном наборе."""
    console.print("[yellow]eval: not yet implemented[/yellow]")


@app.command()
def adapt(
    config: str = typer.Option("configs/adaptation.yaml", "--config"),
    stream: str = typer.Option(..., "--stream"),
    max_steps: int = typer.Option(500, "--max-steps"),
):
    """Онлайн-адаптация по новым данным."""
    console.print("[yellow]adapt: not yet implemented[/yellow]")


if __name__ == "__main__":
    app()
