# 🧬 EDN-Ad — Deepfake Detector

[![CI](https://github.com/SashaEee/diploma_deepfake/actions/workflows/ci.yml/badge.svg)](https://github.com/SashaEee/diploma_deepfake/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

> **English summary:** Video & image deepfake detector — EfficientNet-B0 spatial encoder + adaptive-dilated TCN over time, with a FastAPI/Celery service, a real-time webcam demo, and online domain adaptation. Built in PyTorch.

Система обнаружения дипфейков на видео и изображениях. Покадрово извлекает и выравнивает лицо, кодирует пространственные признаки на **EfficientNet-B0** (FPN + SE), агрегирует их во времени **адаптивным dilated-TCN** и выдаёт вероятность синтеза лица. В комплекте — REST-API на FastAPI с асинхронной обработкой видео через Celery/Redis, веб-демо с камерой и режим онлайн-адаптации к смене домена.

Дипломный проект (А. М. Зенковский, ЮФУ). Лицензия — MIT.

---

## 📌 Что это

**EDN-Ad** = **E**fficientNet **D**ilated-TCN **N**etwork, **Ad**aptive.

Конвейер принимает на вход видеофайл, изображение или поток с камеры. Из кадров детектируется и аффинно выравнивается лицо (MTCNN), затем последовательность из `T = 16` кадров проходит через пространственный и временной энкодеры. На выходе — вероятность того, что лицо является дипфейком, вердикт `real` / `fake` и оценка уверенности.

> ⚠️ **Важно про голову модели.** В коде есть две классификационные «головы»:
> - **`ArcFaceHead`** (угловой margin, `metric_head.py`) — соответствует исходному ТЗ и сохранена для совместимости и тестов;
> - **линейная голова + BCE** (`nn.Linear(embed_dim, 1)`, `full_model.py`) — именно она используется обученной моделью и в инференсе.
>
> Текущий обученный детектор работает с **линейной головой и бинарным логитом**, а не с ArcFace. Это явно отражено в коде (`DeepfakeDetector.head`).

Репозиторий **не содержит** датасетов и поставляет код пайплайна; веса (`best_model.pt`) загружаются отдельно. Часть CLI-команд и интеграционных тестов оставлена заглушками — см. раздел [«Статус»](#-статус).

---

## ✨ Возможности

- 🎞️ **Анализ видео** — потоковое декодирование (PyAV), равномерное семплирование `T` кадров, паддинг при нехватке.
- 🖼️ **Анализ изображений** — одиночное лицо реплицируется на `T` кадров для временного энкодера.
- 📷 **Реальное время с камеры** — скользящий буфер кадров (`deque`), вердикт усредняется по последним результатам.
- 🧠 **Гибридная архитектура** — EfficientNet-B0 + FPN + SE (пространство) и адаптивный dilated-TCN с GLU (время).
- 🔁 **Онлайн-адаптация** — мониторинг доменного дрейфа по KL-расхождению гистограмм активаций и дообучение только `head` + `proj` на уверенных псевдо-метках.
- 🌐 **REST API (FastAPI)** — синхронный разбор изображений, асинхронная обработка видео через Celery + Redis, эндпоинты `/health` и `/metrics` (Prometheus).
- 💻 **Веб-демо** — отдельный сервер `demo_server.py` с тёмным UI, drag-and-drop загрузкой и режимом камеры.
- 🛠️ **CLI (Typer)** — команды `single` и `batch` для обработки файлов.
- 📦 **Docker Compose** — API, Celery-воркер, Redis и Prometheus одной командой.
- 🔍 **Grad-CAM** — модуль визуализации активаций (`inference/gradcam.py`).
- 📤 **Экспорт в ONNX** — скрипт `scripts/export_onnx.py`.

---

## 📊 Результаты

Обучение на **FaceForensics++ (c23)** + **DFDC**, честная **кросс-датасетная** проверка на **CelebDF-v2** (этих данных модель при обучении не видела):

| Датасет | AUROC | F1 | EER |
|---|---|---|---|
| DFDC (валидация, in-domain) | **0.926** | 0.843 | 0.176 |
| CelebDF-v2 (кросс-датасет) | 0.661 | 0.649 | 0.360 |

Кросс-датасетная генерализация — известная сложность детекции дипфейков (другой источник, другие методы синтеза), поэтому метрики на CelebDF-v2 ожидаемо ниже; здесь они приведены честно, без подгонки. Динамика обучения по эпохам — в [`notebooks/training_history.csv`](notebooks/training_history.csv) и `notebooks/results.png`.

## 🏗️ Архитектура

```
Видео / Изображение / Кадр камеры
        │
        ▼
┌──────────────────────────────────────────────┐
│  Препроцессинг                               │
│  VideoLoader (PyAV, T=16) → FaceDetector     │
│  (MTCNN) → аффинное выравнивание к 5 точкам  │
│  (224×224) → FrameNormalizer (ImageNet)      │
└────────────────────┬─────────────────────────┘
                     │ (B, T=16, 3, 224, 224)
                     ▼
┌──────────────────────────────────────────────┐
│  Spatial Encoder                             │
│  EfficientNet-B0 (timm, features_only,       │
│  out_indices 2,3,4) → FPN (3 уровня, 128 ch) │
│  → SE-блоки ×3 → concat → Linear(384→512)+BN │
└────────────────────┬─────────────────────────┘
                     │ (B, T, 512)
                     ▼
┌──────────────────────────────────────────────┐
│  Temporal Encoder (Adaptive Dilated TCN)     │
│  N блоков, каузальные 1D-свёртки + GLU,       │
│  LayerNorm, residual; дилатация выбирается    │
│  динамически по «энергии движения» кадров     │
│  → mean-pooling по времени                   │
└────────────────────┬─────────────────────────┘
                     │ (B, 512)
                     ▼
┌──────────────────────────────────────────────┐
│  Head: Dropout → Linear(512→1)  ⇒  логит     │
│  (ArcFaceHead доступна для совместимости)     │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
   sigmoid → probability ∈ [0,1] → verdict + confidence
```

**Ключевые детали реализации (из кода):**

- **`AdaptiveDilatedConv1d`** — каузальная свёртка, где дилатация `d` пересчитывается на лету из усреднённой «энергии движения» последовательности (`φ = 1/(1+energy)`), затем применяется GLU (`a * sigmoid(b)`).
- **Число слоёв TCN** задаётся конфигом (`base.yaml` → 2, `efficient_tcn.yaml` → 6); базовые дилатации блоков — степени двойки (`1, 2, 4, …`).
- **`shortcut_reg`** — регуляризатор против «срезания углов»: `(mean(|spatial|) − 0.5)²`.
- **`CombinedLoss`** = классификационная (BCE для линейной головы / CE для ArcFace) + `temporal_consistency_loss` (косинусная согласованность соседних эмбеддингов) + `shortcut_reg`.
- **`remap_state_dict`** — переименование ключей чекпоинта, обученного на Kaggle, под имена слоёв проекта.
- **Confidence** считается как `1 − H/ln2`, где `H` — энтропия Бернулли предсказанной вероятности.
- **Адаптация:** `DomainMonitor` копит гистограммы активаций в окне `window`, сравнивает с эталоном симметризованным KL; при `KL > kl_threshold` `AdaptiveTrainer` дообучает только параметры `head`/`proj` на псевдо-метках с `prob > 0.9` или `< 0.1`.

### Формат ответа

```jsonc
// Изображение / видео (CLI, /predict/image, результат Celery-задачи)
{
  "probability": 0.847,   // вероятность дипфейка [0..1]
  "verdict": "fake",      // "fake" если probability >= threshold, иначе "real"; "no_face" если лицо не найдено
  "confidence": 0.71,     // 1 − нормированная энтропия Бернулли
  "face_detected": true,
  "processing_ms": 312
}
```

---

## 🧰 Стек

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-EfficientNet--B0-792EE5)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)
![Celery](https://img.shields.io/badge/Celery-5.3+-37814A?logo=celery&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-broker-DC382D?logo=redis&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-headless-5C3EE8?logo=opencv&logoColor=white)
![PyAV](https://img.shields.io/badge/PyAV-FFmpeg-007808)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![Typer](https://img.shields.io/badge/Typer-CLI-2C2255)
![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?logo=prometheus&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

- **ML:** PyTorch, torchvision, torchmetrics, timm (EfficientNet-B0), facenet-pytorch (MTCNN).
- **Видео/изображения:** PyAV, OpenCV (headless), Pillow, NumPy, SciPy.
- **Сервис:** FastAPI, Uvicorn, Celery, Redis, prometheus-client, python-multipart.
- **Конфиги/CLI:** Pydantic + pydantic-settings, PyYAML, Typer, Rich.
- **Обучение:** scikit-learn, TensorBoard, matplotlib, tqdm.
- **Dev:** pytest (+cov, +asyncio), ruff, mypy, pre-commit; опционально `onnx` / `onnxruntime`.

---

## 🚀 Запуск

### 1. Установка

```bash
git clone https://github.com/SashaEee/diploma_deepfake.git
cd diploma_deepfake

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -e ".[dev]"            # +[onnx] — для экспорта в ONNX
```

Требуется **Python ≥ 3.10**. Для работы с видео в системе нужен **FFmpeg** (в Docker-образ он уже включён).

### 2. Веса модели

Репозиторий не содержит готовых весов в стандартном пути. Положите чекпоинт в `checkpoints/best_model.pt` (путь задаётся в `configs/base.yaml → inference.checkpoint_path`) либо скачайте его:

```bash
# из output Kaggle-ноутбука (нужен настроенный ~/.kaggle/kaggle.json)
python scripts/download_weights.py --source kaggle --kernel <user>/<kernel>

# или по прямой ссылке
python scripts/download_weights.py --source url --url https://example.com/best_model.pt
```

> Если чекпоинт не найден, модель инициализируется случайно (предобучен только backbone EfficientNet-B0 из timm) — предсказания в этом случае не имеют смысла.

### 3. Веб-демо (камера + загрузка изображений)

```bash
python demo_server.py
# открыть http://127.0.0.1:7860
```

### 4. CLI

```bash
# один файл (видео или изображение)
deepfake-cli single --config configs/base.yaml --input sample.mp4
deepfake-cli single --config configs/base.yaml --input face.jpg --output result.json

# пакетная обработка каталога (*.mp4, *.avi)
deepfake-cli batch --config configs/base.yaml --input-dir ./videos/ --output report.json
```

### 5. REST API

```bash
uvicorn deepfake_detector.api.main:app --host 0.0.0.0 --port 8080
```

| Метод и путь                  | Назначение                                              |
|-------------------------------|---------------------------------------------------------|
| `POST /predict/image`         | Синхронный анализ изображения (multipart `file`)        |
| `POST /predict/video`         | Постановка видео в очередь Celery → `{job_id}`           |
| `GET  /predict/video/{job_id}`| Статус и результат задачи (`pending`/`running`/`done`)  |
| `GET  /health`                | Проверка живости и версия модели                        |
| `GET  /metrics`               | Метрики в формате Prometheus                             |

```bash
curl -X POST http://localhost:8080/predict/image -F "file=@face.jpg"

curl -X POST http://localhost:8080/predict/video -F "file=@video.mp4"
# → {"job_id": "...", "estimated_ms": 5000}
curl http://localhost:8080/predict/video/<job_id>
```

> Для эндпоинтов `/predict/video/*` нужен запущенный Redis и Celery-воркер (см. Docker ниже). Брокер берётся из переменной окружения `CELERY_BROKER` (по умолчанию `redis://localhost:6379/0`).

### 6. Docker Compose

Поднимает API, Celery-воркер, Redis и Prometheus:

```bash
docker compose -f docker/docker-compose.yml up -d --build
curl http://localhost:8080/health
```

Образ ожидает веса в `./checkpoints` и конфиги в `./configs` (монтируются как volume).
*Примечание: сервис `prometheus` ссылается на `docker/prometheus.yml` — добавьте этот файл или отключите сервис, если он вам не нужен.*

---

## 🎓 Обучение

```bash
python scripts/train.py \
    --config configs/efficient_tcn.yaml \
    --manifest /data/train_manifest.json \
    --val-manifest /data/val_manifest.json
```

Манифест — JSON-список объектов с полями пути и метки (`0` = real, `1` = fake). Гиперпараметры — в `configs/*.yaml` (эффективный батч = `batch × grad_accum`, mixed precision, веса лоссов `lam_arc/lam_tc/lam_short`).

Экспорт в ONNX:

```bash
python scripts/export_onnx.py --config configs/base.yaml \
    --weights checkpoints/best_model.pt --output deepfake_detector.onnx
```

---

## 🧪 Тесты

```bash
pytest                              # все тесты
pytest tests/unit -v                # только unit
pytest --cov=src/deepfake_detector  # с покрытием
```

Unit-тесты покрывают модели, лоссы, препроцессинг и монитор адаптации. Каталоги `tests/integration` и `tests/smoke` присутствуют как заготовки.

---

## 📁 Структура проекта

```
diploma_deepfake/
├── configs/
│   ├── base.yaml            # инференс по умолчанию (TCN=2 слоя, линейная голова)
│   ├── efficient_tcn.yaml   # обучение (TCN=6, freeze backbone, ArcFace-параметры)
│   └── adaptation.yaml      # параметры онлайн-адаптации
├── docker/
│   ├── Dockerfile           # python:3.10-slim + ffmpeg, editable install
│   └── docker-compose.yml   # api + worker + redis + prometheus
├── scripts/
│   ├── train.py             # точка входа обучения
│   ├── evaluate.py          # оценка (заглушка)
│   ├── download_weights.py  # загрузка весов (Kaggle / URL)
│   └── export_onnx.py       # экспорт в ONNX
├── notebooks/               # Kaggle-ноутбуки обучения, чекпоинты, метрики
├── demo_server.py           # веб-демо (FastAPI + HTML UI, порт 7860)
├── src/deepfake_detector/
│   ├── models/              # spatial_encoder, temporal_encoder, metric_head, full_model
│   ├── preprocessing/       # video_loader, face_detector, aligner, normalizer
│   ├── training/            # trainer, dataset, losses, augmentation
│   ├── adaptation/          # monitor, adaptive_trainer, checkpoint_manager
│   ├── inference/           # predictor, gradcam
│   ├── api/                 # main (FastAPI), tasks (Celery), schemas (Pydantic)
│   ├── cli.py               # Typer CLI
│   └── utils/               # config (Pydantic), seed, logging
└── tests/                   # unit / integration / smoke
```

---

## 📋 Статус

Что реализовано и работает: пространственный и временной энкодеры, препроцессинг видео/лиц, инференс (файл, изображение, камера), FastAPI + Celery-сервис, веб-демо, онлайн-адаптация, экспорт в ONNX, загрузка весов, unit-тесты.

Заглушки / не завершено (по коду): команды CLI `eval` и `adapt` выводят «not yet implemented»; `scripts/evaluate.py` не реализован; каталоги `tests/integration` и `tests/smoke` — заготовки.

---

## 📄 Лицензия

MIT — см. файл [`LICENSE`](LICENSE).
