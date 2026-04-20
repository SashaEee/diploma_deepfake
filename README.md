# EDN-Ad — Deepfake Detector

**EDN-Ad** (EfficientNet-B0 + Dilated TCN + ArcFace — Adaptive) — система обнаружения дипфейков с онлайн-адаптацией к смене домена.

Дипломная работа: Зенковский А. М., ЮФУ

---

## Архитектура

```
Видеофайл / Изображение
       │
       ▼
 ┌─────────────────────────────────────────────┐
 │           Препроцессинг                     │
 │  VideoLoader → FaceDetector (MTCNN)         │
 │  → FaceAligner (аффинное выравнивание)      │
 │  → FrameNormalizer (ImageNet-нормализация)  │
 └───────────────────┬─────────────────────────┘
                     │ (T=16, 3, 224, 224) float32
                     ▼
 ┌─────────────────────────────────────────────┐
 │           Spatial Encoder                   │
 │  EfficientNet-B0 (features_only, stages 2-4)│
 │  → FPN (3 уровня, 128 каналов)              │
 │  → SE-блоки (Squeeze-and-Excitation × 3)   │
 │  → Linear(384 → 512)                        │
 └───────────────────┬─────────────────────────┘
                     │ (B, T=16, D=512)
                     ▼
 ┌─────────────────────────────────────────────┐
 │           Temporal Encoder                  │
 │  Adaptive-Dilated TCN (6 слоёв)             │
 │  Дилатации: 1, 2, 4, 8, 16, 32             │
 │  GLU-активация, LayerNorm, Dropout=0.1      │
 │  Дилатация адаптируется к motion energy     │
 │  → mean-pooling по временной оси            │
 └───────────────────┬─────────────────────────┘
                     │ (B, D=512)
                     ▼
 ┌─────────────────────────────────────────────┐
 │           ArcFace Head                      │
 │  Angular margin: s=32.0, m=0.5              │
 │  2 класса: real / fake                      │
 └───────────────────┬─────────────────────────┘
                     │
                     ▼
              Вероятность deepfake
```

---

## Что подаётся на вход

| Режим           | Вход                                                   | Ограничения               |
|-----------------|--------------------------------------------------------|---------------------------|
| **Видео**       | MP4, MOV, AVI, WebM                                   | до 200 МБ (настраивается) |
| **Изображение** | JPEG, PNG, BMP (RGB, любой размер)                    | —                         |
| **Батч**        | Список путей к видеофайлам                             | —                         |

Система сама извлекает лицо из каждого кадра через MTCNN, выравнивает его аффинным преобразованием к каноническим 5 ориентирам (224×224 пкс) и нормализует по ImageNet-статистике. Если лицо не найдено — кадр пропускается, при нехватке кадров последний повторяется.

---

## Что получается на выходе

```jsonc
// Для видео и изображения:
{
  "probability": 0.847,       // вероятность дипфейка [0.0 — 1.0]
  "verdict": "fake",          // "fake" если prob >= 0.5, иначе "real"
  "confidence": 0.71,         // уверенность модели [0.0 — 1.0] (1 — энтропия)
  "processing_ms": 312        // время обработки в миллисекундах
}
```

Через REST API асинхронное видео возвращает `job_id`, статус которого опрашивается отдельно:

```jsonc
// GET /predict/video/{job_id}
{
  "job_id": "uuid-...",
  "status": "done",          // "pending" | "done" | "error"
  "probability": 0.91,
  "verdict": "fake",
  "confidence": 0.84,
  "processing_ms": 1240
}
```

---

## Датасеты

Собственных данных проект **не включает** — нужны лицензионные датасеты. Система обучается на публичных бенчмарках:

| Датасет                  | Размер             | Назначение              | Доступ              |
|--------------------------|--------------------|-------------------------|---------------------|
| **FaceForensics++**      | ~1.5 ТБ (raw)      | Обучение + валидация    | [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) — нужен запрос |
| **DFDC**                 | ~470 ГБ            | Cross-датасет тест      | Kaggle              |
| **CelebDF-v2**           | ~3 ГБ              | Cross-датасет тест      | [github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics) |

**FaceForensics++** содержит 4 типа манипуляций: `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures` в трёх качествах (raw, c23, c40). Система обучалась на сжатии c23.

### Формат манифеста для обучения

```json
[
  {"path": "/data/ff++/fake/Deepfakes/001_003.mp4", "label": 1, "manipulation": "Deepfakes"},
  {"path": "/data/ff++/real/original/001.mp4",      "label": 0, "manipulation": "none"}
]
```

---

## Установка

```bash
# Python 3.10+
git clone <repo>
cd deepfake-detector

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -e ".[dev]"
```

---

## Обучение

```bash
python scripts/train.py \
    --config configs/base.yaml \
    --manifest /data/ff++/train_manifest.json \
    --val-manifest /data/ff++/val_manifest.json
```

Гиперпараметры по умолчанию (`configs/base.yaml`):

| Параметр            | Значение      | Описание                            |
|---------------------|---------------|-------------------------------------|
| `embed_dim`         | 512           | Размер пространства признаков       |
| `n_layers`          | 6             | Число блоков TCN                    |
| `arc_s / arc_m`     | 32.0 / 0.5    | Масштаб и margin ArcFace            |
| `batch`             | 4             | Физический батч                     |
| `grad_accum`        | 16            | Накопление → эффективный батч = 64  |
| `lr`                | 3e-4          | AdamW learning rate                 |
| `epochs`            | 60            | Cosine annealing                    |
| `mixed`             | true          | BF16 mixed precision                |

---

## Инференс

### CLI

```bash
# Одно видео
deepfake-cli single --config configs/base.yaml --input sample.mp4

# Батч файлов
deepfake-cli batch --config configs/base.yaml \
    --input-dir ./videos/ --output report.json

# Оценка на датасете
deepfake-cli eval --config configs/base.yaml \
    --manifest ff++_val.json --metrics auroc,f1,eer

# Онлайн-адаптация к новому домену
deepfake-cli adapt --config configs/adaptation.yaml --stream ./new_data/
```

### REST API

```bash
uvicorn deepfake_detector.api.main:app --host 0.0.0.0 --port 8080

# Изображение (синхронно, ~300 мс)
curl -X POST http://localhost:8080/predict/image \
     -F "file=@face.jpg"

# Видео (асинхронно через Celery + Redis)
curl -X POST http://localhost:8080/predict/video \
     -F "file=@video.mp4"
# → {"job_id": "...", "estimated_ms": 5000}

curl http://localhost:8080/predict/video/{job_id}
# → {"status": "done", "probability": 0.91, ...}

# Здоровье сервиса
curl http://localhost:8080/health
```

### Docker

```bash
docker compose -f docker/docker-compose.yml up -d
curl http://localhost:8080/health
```

---

## Онлайн-адаптация

`DomainMonitor` отслеживает KL-дивергенцию гистограмм активаций от опорного распределения. При обнаружении дрейфа домена (`KL > 0.35`) `AdaptiveTrainer` дообучает только `head` и `proj` (~0.03% параметров) на псевдо-метках с высокой уверенностью (> 0.9).

```
configs/adaptation.yaml:
  adapt_lr: 1e-5
  confidence_high: 0.9
  window: 1000          # сколько последних активаций хранить
  kl_threshold: 0.35
```

---

## Тесты

```bash
# Все unit-тесты (134 шт., ~5 сек)
pytest tests/unit/ -v

# С покрытием
pytest tests/unit/ --cov=src/deepfake_detector --cov-report=term-missing
```

Покрытие ключевых модулей:

| Модуль                  | Покрытие |
|-------------------------|----------|
| `video_loader.py`       | 100 %    |
| `face_detector.py`      | 100 %    |
| `losses.py`             | 100 %    |
| `monitor.py`            | 100 %    |
| `spatial_encoder.py`    | 100 %    |
| `temporal_encoder.py`   | 100 %    |
| `metric_head.py`        | 100 %    |

---

## Структура проекта

```
deepfake-detector/
├── configs/
│   ├── base.yaml           # основные гиперпараметры
│   └── adaptation.yaml     # параметры онлайн-адаптации
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── export_onnx.py
├── src/deepfake_detector/
│   ├── models/             # SpatialEncoder, TemporalEncoder, ArcFaceHead, DeepfakeDetector
│   ├── preprocessing/      # VideoLoader, FaceDetector, FaceAligner, FrameNormalizer
│   ├── training/           # Trainer, CombinedLoss, VideoDataset, Augmentation
│   ├── adaptation/         # DomainMonitor, AdaptiveTrainer, CheckpointManager
│   ├── inference/          # Predictor, GradCAM
│   ├── api/                # FastAPI, Celery tasks, Pydantic schemas
│   ├── cli.py              # Typer CLI
│   └── utils/              # config, seed, logging
└── tests/
    ├── unit/               # 134 теста — все зелёные
    ├── integration/        # не реализованы (заглушки)
    └── smoke/              # не реализованы (заглушки)
```

---

## Зависимости

- Python ≥ 3.10
- PyTorch ≥ 2.2
- timm (EfficientNet-B0)
- facenet-pytorch (MTCNN)
- OpenCV, PyAV, NumPy
- FastAPI, Celery, Redis
- Typer (CLI)
