"""
Demo-сервер для тестирования Deepfake Detector.
Запуск: python3.12 demo_server.py
Браузер: http://127.0.0.1:7860
"""
from __future__ import annotations

import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    from deepfake_detector.inference.predictor import Predictor
    predictor = Predictor.from_config("configs/base.yaml")
    print("[demo] Model loaded.")
    yield


app = FastAPI(lifespan=lifespan)


class FrameReq(BaseModel):
    image: str  # base64 JPEG/PNG
    mode: str = "image"  # "image" or "camera"


@app.post("/analyze")
async def analyze(req: FrameReq):
    if predictor is None:
        return {"probability": 0.5, "verdict": "loading", "confidence": 0.0,
                "face_detected": False}
    data = base64.b64decode(req.image)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"probability": 0.5, "verdict": "error", "confidence": 0.0,
                "face_detected": False}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if req.mode == "camera":
        return predictor.predict_camera_frame(img)
    return predictor.predict_image(img)


@app.post("/reset")
async def reset():
    """Сброс буфера кадров камеры."""
    if predictor is not None:
        predictor.reset_buffer()
    return {"status": "ok"}


@app.get("/")
async def index():
    return HTMLResponse(HTML)


# ──────────────────────────────────────────────────────────────────────────────
# HTML / CSS / JS — Apple-inspired dark UI
# ──────────────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deepfake Detector</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #000000;
    --surface: rgba(255,255,255,0.055);
    --border:  rgba(255,255,255,0.10);
    --text:    #f5f5f7;
    --muted:   rgba(255,255,255,0.45);
    --dim:     rgba(255,255,255,0.18);
    --real:    #30d158;
    --fake:    #ff453a;
    --blue:    #0a84ff;
    --yellow:  #ffd60a;
    --r:       18px;
    --r-sm:    10px;
  }

  html, body {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }

  body {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 20px 60px;
    min-height: 100%;
  }

  /* ── Header ─────────────────────────────────── */
  header {
    width: 100%;
    max-width: 700px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 36px;
  }

  .wordmark {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .wordmark svg { flex-shrink: 0; }

  .wordmark-text {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .status-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: var(--muted);
    text-transform: uppercase;
  }

  .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--dim);
    transition: background 0.4s;
  }
  .dot.live  { background: var(--real);  animation: blink 1.8s ease-in-out infinite; }
  .dot.error { background: var(--fake); }
  .dot.wait  { background: var(--blue); animation: blink 1.8s ease-in-out infinite; }
  .dot.warn  { background: var(--yellow); animation: blink 1.2s ease-in-out infinite; }

  @keyframes blink {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.35; }
  }

  /* ── Camera wrap ─────────────────────────────── */
  .camera-wrap {
    position: relative;
    width: 100%;
    max-width: 700px;
    aspect-ratio: 16/9;
    background: #0a0a0a;
    border-radius: var(--r);
    overflow: hidden;
    border: 1px solid var(--border);
  }

  video {
    width: 100%; height: 100%;
    object-fit: cover;
    display: block;
  }

  canvas#cap { display: none; }

  /* Drop zone (shown when camera is off) */
  .drop-zone {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    cursor: pointer;
    transition: background 0.2s;
  }
  .drop-zone:hover { background: rgba(255,255,255,0.03); }
  .drop-zone.hidden { display: none; }
  .drop-zone input { display: none; }

  .drop-icon {
    width: 52px; height: 52px;
    border: 1.5px solid var(--dim);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    color: var(--dim);
  }

  .drop-label {
    font-size: 14px;
    color: var(--muted);
    font-weight: 400;
  }
  .drop-label span {
    color: var(--blue);
    font-weight: 500;
    cursor: pointer;
  }

  /* Corner brackets (always visible) */
  .corner {
    position: absolute;
    width: 18px; height: 18px;
    border-color: rgba(255,255,255,0.28);
    border-style: solid;
    pointer-events: none;
  }
  .c-tl { top:12px; left:12px;  border-width: 1.5px 0 0 1.5px; }
  .c-tr { top:12px; right:12px; border-width: 1.5px 1.5px 0 0; }
  .c-bl { bottom:12px; left:12px;  border-width: 0 0 1.5px 1.5px; }
  .c-br { bottom:12px; right:12px; border-width: 0 1.5px 1.5px 0; }

  /* Scan line */
  .scan {
    position: absolute;
    left: 0; right: 0;
    height: 1.5px;
    background: linear-gradient(90deg, transparent 0%, var(--blue) 50%, transparent 100%);
    top: -4px;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s;
  }
  .scan.active {
    opacity: 0.7;
    animation: scanMove 2.2s cubic-bezier(0.4,0,0.6,1) infinite;
  }

  @keyframes scanMove {
    0%   { top: 0%; }
    100% { top: 100%; }
  }

  /* Verdict badge on camera */
  .badge {
    position: absolute;
    bottom: 16px; left: 50%;
    transform: translateX(-50%) translateY(6px);
    background: rgba(0,0,0,0.72);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 100px;
    padding: 7px 18px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    white-space: nowrap;
    opacity: 0;
    transition: opacity 0.35s ease, transform 0.35s ease, border-color 0.35s, color 0.35s;
  }
  .badge.show {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }

  /* Buffer indicator */
  .buffer-bar {
    position: absolute;
    top: 12px; right: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,0,0,0.60);
    backdrop-filter: blur(12px);
    border-radius: 8px;
    padding: 5px 10px;
    font-size: 10px;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: 0.04em;
    opacity: 0;
    transition: opacity 0.3s;
  }
  .buffer-bar.show { opacity: 1; }
  .buffer-dots {
    display: flex;
    gap: 2px;
  }
  .buffer-dots .bd {
    width: 4px; height: 10px;
    border-radius: 1px;
    background: rgba(255,255,255,0.12);
    transition: background 0.3s;
  }
  .buffer-dots .bd.filled {
    background: var(--blue);
  }

  /* Preview image (when file dropped) */
  .preview-img {
    position: absolute;
    inset: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    display: none;
  }
  .preview-img.show { display: block; }

  /* ── Results panel ───────────────────────────── */
  .panel {
    width: 100%;
    max-width: 700px;
    margin-top: 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 30px 32px 28px;
  }

  .verdict-row {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 4px;
  }

  .verdict-text {
    font-size: 52px;
    font-weight: 700;
    letter-spacing: -1.5px;
    line-height: 1;
    color: var(--dim);
    transition: color 0.4s ease;
  }
  .verdict-text.real    { color: var(--real); }
  .verdict-text.fake    { color: var(--fake); }
  .verdict-text.no-face { color: var(--yellow); }

  .verdict-prob {
    font-size: 28px;
    font-weight: 300;
    letter-spacing: -0.5px;
    color: var(--muted);
    padding-bottom: 4px;
    transition: color 0.4s;
    font-variant-numeric: tabular-nums;
  }

  .verdict-sub {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 26px;
    height: 18px;
    transition: color 0.4s;
  }

  /* Bars */
  .bars { display: flex; flex-direction: column; gap: 12px; }

  .bar-row {}
  .bar-meta {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    letter-spacing: 0.03em;
    color: var(--muted);
    margin-bottom: 5px;
  }
  .bar-track {
    height: 2px;
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 100px;
    width: 0%;
    transition: width 0.55s cubic-bezier(0.4,0,0.2,1), background 0.4s;
  }

  /* Stats row */
  .stats {
    display: flex;
    gap: 0;
    margin-top: 26px;
    padding-top: 22px;
    border-top: 1px solid var(--border);
  }

  .stat {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .stat + .stat {
    padding-left: 28px;
    border-left: 1px solid var(--border);
    margin-left: 28px;
  }

  .stat-val {
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.5px;
    font-variant-numeric: tabular-nums;
    color: var(--text);
  }
  .stat-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
  }

  /* ── Buttons ─────────────────────────────────── */
  .btn-row {
    display: flex;
    gap: 10px;
    margin-top: 22px;
  }

  .btn {
    border: none;
    border-radius: 100px;
    padding: 11px 28px;
    font-size: 14px;
    font-weight: 600;
    font-family: inherit;
    cursor: pointer;
    letter-spacing: -0.01em;
    transition: opacity 0.18s, background 0.2s;
  }
  .btn:active { opacity: 0.6; }

  .btn-primary { background: var(--blue); color: #fff; }
  .btn-primary:hover { opacity: 0.88; }

  .btn-secondary {
    background: rgba(255,255,255,0.08);
    color: var(--text);
    border: 1px solid var(--border);
  }
  .btn-secondary:hover { background: rgba(255,255,255,0.12); }

  .btn-danger { background: rgba(255,69,58,0.18); color: var(--fake); border: 1px solid rgba(255,69,58,0.28); }
  .btn-danger:hover { background: rgba(255,69,58,0.25); }

  .btn:disabled { opacity: 0.3; cursor: default; }

  /* ── Misc ────────────────────────────────────── */
  .hidden { display: none !important; }
</style>
</head>
<body>

<!-- Header -->
<header>
  <div class="wordmark">
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <circle cx="10" cy="10" r="9" stroke="rgba(255,255,255,0.25)" stroke-width="1.2"/>
      <circle cx="10" cy="10" r="4" fill="rgba(255,255,255,0.15)"/>
      <circle cx="10" cy="10" r="1.5" fill="rgba(255,255,255,0.7)"/>
    </svg>
    <span class="wordmark-text">Deepfake Detector</span>
  </div>
  <div class="status-pill">
    <div class="dot" id="dot"></div>
    <span id="statusText">Готов</span>
  </div>
</header>

<!-- Camera / Preview -->
<div class="camera-wrap" id="cameraWrap">
  <video id="video" autoplay muted playsinline></video>
  <canvas id="cap"></canvas>

  <!-- Drop zone -->
  <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileIn').click()">
    <input type="file" id="fileIn" accept="image/*,video/*">
    <div class="drop-icon">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="17 8 12 3 7 8"/>
        <line x1="12" y1="3" x2="12" y2="15"/>
      </svg>
    </div>
    <div class="drop-label">Перетащите файл или <span>выберите</span></div>
  </div>

  <!-- File preview (image) -->
  <img class="preview-img" id="previewImg" alt="">

  <!-- Buffer indicator (camera mode) -->
  <div class="buffer-bar" id="bufferBar">
    <span>БУФЕР</span>
    <div class="buffer-dots" id="bufferDots"></div>
    <span id="bufferLabel">0/16</span>
  </div>

  <!-- Scan line -->
  <div class="scan" id="scan"></div>

  <!-- Corner brackets -->
  <div class="corner c-tl"></div>
  <div class="corner c-tr"></div>
  <div class="corner c-bl"></div>
  <div class="corner c-br"></div>

  <!-- Verdict badge -->
  <div class="badge" id="badge">&mdash;</div>
</div>

<!-- Results panel -->
<div class="panel">
  <div class="verdict-row">
    <div class="verdict-text" id="vText">&mdash;</div>
    <div class="verdict-prob" id="vProb"></div>
  </div>
  <div class="verdict-sub" id="vSub">Запустите камеру или загрузите изображение</div>

  <div class="bars">
    <div class="bar-row">
      <div class="bar-meta">
        <span>Вероятность фейка</span>
        <span id="probPct">&mdash;</span>
      </div>
      <div class="bar-track"><div class="bar-fill" id="probFill"></div></div>
    </div>
    <div class="bar-row">
      <div class="bar-meta">
        <span>Уверенность</span>
        <span id="confPct">&mdash;</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" id="confFill" style="background:rgba(255,255,255,0.3)"></div>
      </div>
    </div>
  </div>

  <div class="stats">
    <div class="stat">
      <span class="stat-val" id="sLat">&mdash;</span>
      <span class="stat-label">Задержка</span>
    </div>
    <div class="stat">
      <span class="stat-val" id="sFrames">0</span>
      <span class="stat-label">Кадров</span>
    </div>
    <div class="stat">
      <span class="stat-val">0.75</span>
      <span class="stat-label">AUROC (DFDC)</span>
    </div>
    <div class="stat">
      <span class="stat-val">EDN-Ad</span>
      <span class="stat-label">Модель</span>
    </div>
  </div>

  <div class="btn-row">
    <button class="btn btn-primary" id="btnCam" onclick="toggleCamera()">Камера</button>
    <button class="btn btn-secondary" onclick="document.getElementById('fileIn').click()">Загрузить файл</button>
    <button class="btn btn-danger hidden" id="btnStop" onclick="stopAll()">Стоп</button>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);

let stream = null, camInterval = null, frames = 0, running = false, mode = 'idle';
let probHistory = [];
const BUFFER_SIZE = 16;

/* ── Buffer dots init ──────────────────────────── */
(function initBufferDots() {
  const wrap = $('bufferDots');
  for (let i = 0; i < BUFFER_SIZE; i++) {
    const d = document.createElement('div');
    d.className = 'bd';
    wrap.appendChild(d);
  }
})();

function updateBufferUI(filled) {
  const dots = $('bufferDots').children;
  for (let i = 0; i < BUFFER_SIZE; i++) {
    dots[i].className = i < filled ? 'bd filled' : 'bd';
  }
  $('bufferLabel').textContent = Math.min(filled, BUFFER_SIZE) + '/' + BUFFER_SIZE;
}

/* ── Status helpers ──────────────────────────── */
function setStatus(text, dotClass) {
  $('statusText').textContent = text;
  $('dot').className = 'dot ' + dotClass;
}

/* ── Camera ──────────────────────────────────── */
async function toggleCamera() {
  if (running && mode === 'camera') { stopAll(); return; }
  stopAll(true);

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    $('video').srcObject = stream;
    $('video').style.display = 'block';
    $('dropZone').classList.add('hidden');
    $('previewImg').classList.remove('show');
    $('scan').classList.add('active');
    $('bufferBar').classList.add('show');
    $('btnCam').textContent = 'Выключить';
    $('btnStop').classList.remove('hidden');
    running = true; mode = 'camera';
    setStatus('Анализ…', 'live');
    camInterval = setInterval(analyzeFrame, 900);
  } catch(e) {
    setStatus('Нет доступа к камере', 'error');
  }
}

async function analyzeFrame() {
  const video = $('video');
  if (!video.readyState || video.readyState < 2) return;
  const c = $('cap');
  c.width = video.videoWidth || 640;
  c.height = video.videoHeight || 360;
  c.getContext('2d').drawImage(video, 0, 0);
  const b64 = c.toDataURL('image/jpeg', 0.82).split(',')[1];
  await sendImage(b64, 'camera');
}

/* ── File upload ─────────────────────────────── */
$('fileIn').addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  e.target.value = '';

  if (file.type.startsWith('image/')) {
    stopAll(true);
    const url = URL.createObjectURL(file);
    $('previewImg').src = url;
    $('previewImg').classList.add('show');
    $('dropZone').classList.add('hidden');
    $('video').style.display = 'none';
    $('scan').classList.add('active');
    setStatus('Анализ…', 'wait');
    mode = 'image';

    const reader = new FileReader();
    reader.onload = async ev => {
      const b64 = ev.target.result.split(',')[1];
      await sendImage(b64, 'image');
      $('scan').classList.remove('active');
      setStatus('Готово', 'live');
    };
    reader.readAsDataURL(file);
  } else {
    alert('Для видео используйте камеру. Загрузите изображение (.jpg, .png).');
  }
});

/* Drag & drop */
const wrap = $('cameraWrap');
wrap.addEventListener('dragover', e => { e.preventDefault(); });
wrap.addEventListener('drop', e => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f) { $('fileIn').files = e.dataTransfer.files; $('fileIn').dispatchEvent(new Event('change')); }
});

/* ── Core: send frame to server ──────────────── */
async function sendImage(b64, sendMode) {
  const t0 = Date.now();
  try {
    const r = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64, mode: sendMode || 'image' })
    });
    const d = await r.json();
    frames++;
    updateUI(d, Date.now() - t0);
  } catch(err) {
    setStatus('Ошибка сервера', 'error');
  }
}

/* ── Update UI ───────────────────────────────── */
function updateUI(d, ms) {
  const faceDetected = d.face_detected !== false;
  const bufFrames = d.buffer_frames || 0;

  // Update buffer indicator (camera mode)
  if (mode === 'camera') {
    updateBufferUI(bufFrames);
  }

  // No face detected
  if (!faceDetected) {
    $('vText').textContent = 'Нет лица';
    $('vText').className = 'verdict-text no-face';
    $('vProb').textContent = '';
    $('vSub').textContent = 'Лицо не обнаружено — направьте камеру на лицо';
    if (mode === 'camera') {
      $('vSub').textContent = bufFrames > 0
        ? 'Лицо потеряно — используется буфер (' + bufFrames + ' кадров)'
        : 'Лицо не обнаружено — направьте камеру на лицо';
    }

    const badge = $('badge');
    badge.textContent = 'НЕТ ЛИЦА';
    badge.style.color = '#ffd60a';
    badge.style.borderColor = 'rgba(255,214,10,0.35)';
    badge.classList.add('show');

    setStatus('Нет лица', 'warn');

    $('probFill').style.width = '0%';
    $('probPct').textContent = '—';
    $('confFill').style.width = '0%';
    $('confPct').textContent = '—';
    $('sLat').textContent = ms + ' ms';
    $('sFrames').textContent = frames;

    // If buffer has frames, still show the model result
    if (bufFrames > 0 && d.probability !== undefined && d.probability !== 0.5) {
      showResult(d, ms);
    }
    return;
  }

  showResult(d, ms);
}

function showResult(d, ms) {
  let prob = d.probability || 0;

  // Sliding average for camera (over last 5 results)
  if (mode === 'camera') {
    probHistory.push(prob);
    if (probHistory.length > 5) probHistory.shift();
    prob = probHistory.reduce((a, b) => a + b, 0) / probHistory.length;
  }

  const conf    = d.confidence || 0;
  const uncertain = conf < 0.15;
  const isFake  = !uncertain && prob >= 0.5;
  const color   = uncertain ? 'var(--muted)' : isFake ? 'var(--fake)' : 'var(--real)';
  const label   = uncertain ? 'Неизвестно' : isFake ? 'Deepfake' : 'Настоящее';
  const probPct = Math.round(prob * 100);
  const confPct = Math.round(conf * 100);
  const cls     = uncertain ? '' : isFake ? 'fake' : 'real';

  /* Verdict */
  const vt = $('vText');
  vt.textContent = label;
  vt.className = 'verdict-text ' + cls;

  $('vProb').textContent = probPct + '%';
  $('vProb').style.color = color;

  $('vSub').textContent = uncertain
    ? 'Уверенность модели слишком низкая для вердикта'
    : isFake
      ? 'Обнаружены артефакты синтеза лица'
      : 'Признаков манипуляции не обнаружено';

  /* Badge */
  const badge = $('badge');
  badge.textContent = label.toUpperCase() + ' · ' + probPct + '%';
  badge.style.color = isFake ? '#ff453a' : '#30d158';
  badge.style.borderColor = isFake ? 'rgba(255,69,58,0.35)' : 'rgba(48,209,88,0.35)';
  badge.classList.add('show');

  if (d.face_detected !== false) {
    setStatus('Анализ…', 'live');
  }

  /* Bars */
  const pf = $('probFill');
  pf.style.width = probPct + '%';
  pf.style.background = color;
  $('probPct').textContent = probPct + '%';

  $('confFill').style.width = confPct + '%';
  $('confPct').textContent = confPct + '%';

  /* Stats */
  $('sLat').textContent = ms + ' ms';
  $('sFrames').textContent = frames;
}

/* ── Stop everything ─────────────────────────── */
async function stopAll(silent = false) {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  clearInterval(camInterval); camInterval = null;
  $('video').srcObject = null;
  $('video').style.display = 'none';
  $('scan').classList.remove('active');
  $('bufferBar').classList.remove('show');
  $('dropZone').classList.remove('hidden');
  $('btnCam').textContent = 'Камера';
  $('btnStop').classList.add('hidden');
  running = false; mode = 'idle';
  probHistory = [];

  // Reset server-side buffer
  try { await fetch('/reset', { method: 'POST' }); } catch(e) {}

  if (!silent) {
    resetUI();
    setStatus('Готов', '');
  }
}

function resetUI() {
  $('vText').textContent = '—';
  $('vText').className = 'verdict-text';
  $('vProb').textContent = '';
  $('vSub').textContent = 'Запустите камеру или загрузите изображение';
  $('badge').classList.remove('show');
  $('probFill').style.width = '0%';
  $('confFill').style.width = '0%';
  $('probPct').textContent = '—';
  $('confPct').textContent = '—';
  $('sLat').textContent = '—';
  $('sFrames').textContent = frames = 0;
  updateBufferUI(0);
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(
        "demo_server:app",
        host="127.0.0.1",
        port=7860,
        reload=False,
        log_level="warning",
    )
