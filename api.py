"""REST API для транскрибирования аудио.

Запуск:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Эндпоинты:
    POST /transcribe          — загрузить файл и запустить обработку
    GET  /status/{task_id}    — статус задачи
    GET  /result/{task_id}    — получить результат (plain/srt/json)
    GET  /health              — проверка работоспособности
"""
import asyncio
import json
import os
import tempfile
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional

import aiofiles
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

import postprocess
from config import Config

# ══════════════════════════════════════════════════════════════
#  Глобальные объекты (загружаются один раз при старте)
# ══════════════════════════════════════════════════════════════

cfg = Config()
_model = None
_processor = None
_pipe = None


def _load_model():
    global _model, _processor, _pipe
    if _model is not None:
        return

    print("Загрузка модели…")
    base = WhisperForConditionalGeneration.from_pretrained(
        cfg.MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    if Path(cfg.OUTPUT_DIR).exists():
        _model = PeftModel.from_pretrained(base, cfg.OUTPUT_DIR)
        print(f"LoRA-адаптеры загружены из {cfg.OUTPUT_DIR}")
    else:
        _model = base
        print("LoRA не найден — используется базовая модель")

    _model.eval()
    _processor = WhisperProcessor.from_pretrained(
        cfg.MODEL_NAME, language=cfg.LANGUAGE, task=cfg.TASK
    )
    print("Модель готова.")


# ══════════════════════════════════════════════════════════════
#  Хранилище задач (in-memory)
# ══════════════════════════════════════════════════════════════

class TaskStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    ERROR      = "error"


tasks: dict[str, dict] = {}  # task_id → {status, result, error, created_at}


# ══════════════════════════════════════════════════════════════
#  Вспомогательные функции
# ══════════════════════════════════════════════════════════════

def _chunks_to_srt(chunks: list) -> str:
    """Конвертирует Whisper-chunks (с тайм-кодами) в формат SRT."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        ts = chunk.get("timestamp", (0.0, 0.0))
        start_s, end_s = (ts[0] or 0.0), (ts[1] or ts[0] or 0.0)

        def fmt(s):
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sc = s % 60
            return f"{h:02d}:{m:02d}:{sc:06.3f}".replace(".", ",")

        lines.append(str(i))
        lines.append(f"{fmt(start_s)} --> {fmt(end_s)}")
        lines.append(chunk["text"].strip())
        lines.append("")

    return "\n".join(lines)


async def _process_task(task_id: str, audio_path: str, run_postprocess: bool):
    """Фоновая задача транскрибирования."""
    import traceback as _tb
    try:
        tasks[task_id]["status"] = TaskStatus.PROCESSING

        import torchaudio as _ta
        import torchaudio.transforms as _T

        # ── Загрузка аудио (torchaudio, без torchcodec) ───────
        waveform, sr = _ta.load(audio_path)
        if sr != 16000:
            waveform = _T.Resample(sr, 16000)(waveform)
        arr = waveform.mean(0).numpy()          # (T,) float32, 16 kHz
        total_sec = len(arr) / 16000

        # ── Прямой инференс (без pipeline, чтобы обойти torchcodec) ──
        dtype  = next(_model.parameters()).dtype
        CHUNK  = 30 * 16000   # 480 000 сэмплов = 30 с
        STRIDE =  5 * 16000   # 80 000 сэмплов  = 5 с перекрытие

        chunks: list = []

        if len(arr) <= CHUNK:
            # Короткое аудио — один проход
            feats = _processor(arr, sampling_rate=16000,
                               return_tensors="pt").input_features
            feats = feats.to(_model.device, dtype=dtype)
            with torch.no_grad():
                ids = _model.generate(feats, language=cfg.LANGUAGE,
                                      task=cfg.TASK, max_new_tokens=225)
            text = _processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            chunks = [{"text": text, "timestamp": (0.0, total_sec)}]
        else:
            # Длинное аудио — чанки с перекрытием
            pos = 0
            while pos < len(arr):
                end       = min(pos + CHUNK, len(arr))
                feats     = _processor(arr[pos:end], sampling_rate=16000,
                                       return_tensors="pt").input_features
                feats     = feats.to(_model.device, dtype=dtype)
                with torch.no_grad():
                    ids = _model.generate(feats, language=cfg.LANGUAGE,
                                          task=cfg.TASK, max_new_tokens=225)
                text = _processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
                chunks.append({"text": text,
                               "timestamp": (pos / 16000, end / 16000)})
                pos += CHUNK - STRIDE

        raw_text = " ".join(c["text"] for c in chunks)

        # ── Постобработка ─────────────────────────────────────
        if run_postprocess:
            pp         = postprocess.process(raw_text,
                             restore_punct=False, normalize_nums=False)
            final_text = pp.text_final
            entities   = [e.__dict__ for e in pp.entities]
        else:
            final_text = raw_text
            entities   = []

        srt = _chunks_to_srt(chunks)

        tasks[task_id].update({
            "status":      TaskStatus.DONE,
            "text":        final_text,
            "text_raw":    raw_text,
            "srt":         srt,
            "chunks":      chunks,
            "entities":    entities,
            "finished_at": time.time(),
        })

    except Exception as e:
        tasks[task_id].update({
            "status": TaskStatus.ERROR,
            "error":  f"{e}\n\n{_tb.format_exc()}",
        })
    finally:
        # удаляем временный файл
        try:
            os.unlink(audio_path)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
#  FastAPI приложение
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Whisper ASR API",
    description="REST API для транскрибирования аудио с постобработкой",
    version="1.0.0",
)


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)


@app.get("/health")
def health():
    return {"status": "ok", "model": cfg.MODEL_NAME, "lora": Path(cfg.OUTPUT_DIR).exists()}


@app.post("/transcribe", status_code=202)
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    postprocessing: bool = Query(True, description="Применить постобработку"),
):
    """Загрузить аудиофайл и запустить транскрибирование."""
    allowed = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Формат {suffix} не поддерживается. Допустимые: {allowed}")

    # Сохраняем во временный файл
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    async with aiofiles.open(tmp.name, "wb") as f:
        await f.write(await file.read())
    tmp.close()

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status":     TaskStatus.PENDING,
        "filename":   file.filename,
        "created_at": time.time(),
    }

    background_tasks.add_task(_process_task, task_id, tmp.name, postprocessing)
    return {"task_id": task_id, "status": TaskStatus.PENDING}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    """Проверить статус задачи."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Задача не найдена")
    return {
        "task_id":    task_id,
        "status":     task["status"],
        "filename":   task.get("filename"),
        "created_at": task.get("created_at"),
        "error":      task.get("error"),
    }


@app.get("/result/{task_id}")
def get_result(
    task_id: str,
    format: str = Query("json", description="Формат: json | text | srt"),
):
    """Получить результат транскрибирования."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Задача не найдена")
    if task["status"] == TaskStatus.PROCESSING:
        raise HTTPException(202, "Задача ещё выполняется")
    if task["status"] == TaskStatus.ERROR:
        raise HTTPException(500, f"Ошибка: {task.get('error')}")
    if task["status"] != TaskStatus.DONE:
        raise HTTPException(400, "Задача не завершена")

    if format == "text":
        return PlainTextResponse(task["text"])

    if format == "srt":
        return PlainTextResponse(task["srt"], media_type="text/plain; charset=utf-8")

    # JSON (по умолчанию)
    return {
        "task_id":  task_id,
        "filename": task.get("filename"),
        "text":     task["text"],
        "text_raw": task.get("text_raw", task["text"]),
        "entities": task.get("entities", []),
        "chunks":   task.get("chunks", []),
        "duration": (task.get("finished_at", 0) - task.get("created_at", 0)),
    }


@app.get("/tasks")
def list_tasks():
    """Список всех задач."""
    return [
        {"task_id": tid, "status": t["status"], "filename": t.get("filename")}
        for tid, t in tasks.items()
    ]
