"""
Оценка четырёх ASR-систем на Multilingual TEDx Russian (локальный датасет).
Формальный стиль речи: TED-выступления, профессиональная дикция.

Структура датасета (C:\\whisper-finetune\\media\\ru-ru\\data\\test):
    txt/test.yaml  — сегменты: {duration, offset, speaker_id, wav}
    txt/test.ru    — транскрипции (одна строка на сегмент)
    wav/           — полные FLAC-файлы (извлечение по offset+duration)

Запуск:
    python eval_tedx.py                        # все 4 модели
    python eval_tedx.py --models baseline ft   # выборочно
    python eval_tedx.py --max_samples 200      # быстрый тест

Результаты сохраняются в:
    predictions_tedx_baseline.txt
    predictions_tedx_ft.txt
    predictions_tedx_gigaam.txt
    predictions_tedx_borealis.txt
    results/tedx_comparison.csv
    results/tedx_wer.png
"""
import argparse
import os
import re
import tempfile
import time

import numpy as np
import torch
import soundfile as sf
import yaml
import evaluate
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["font.family"] = "DejaVu Sans"

from config import Config
import postprocess

# ══════════════════════════════════════════════════════════════
#  Нормализация (удаление пунктуации, нижний регистр)
# ══════════════════════════════════════════════════════════════

_wer_metric  = evaluate.load("wer")
_paren_re    = re.compile(r"\([^)]*\)")          # (Аплодисменты), (Смех) и т.п.
_punct_re    = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    text = _paren_re.sub("", text)               # убираем скобочные аннотации
    text = text.lower().strip()
    text = _punct_re.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_wer(refs, preds):
    refs_n  = [normalize(r) for r in refs]
    preds_n = [normalize(p) for p in preds]
    return _wer_metric.compute(predictions=preds_n, references=refs_n) * 100


# ══════════════════════════════════════════════════════════════
#  Загрузка датасета из локальной директории
# ══════════════════════════════════════════════════════════════

DATASET_ROOT = r"C:\whisper-finetune\media\ru-ru\data\test"


def load_tedx(max_samples: int | None = None):
    """
    Возвращает список словарей:
        {"audio": np.ndarray float32 16kHz, "transcription": str}
    """
    txt_dir = os.path.join(DATASET_ROOT, "txt")
    wav_dir = os.path.join(DATASET_ROOT, "wav")

    yaml_path = os.path.join(txt_dir, "test.yaml")
    ref_path  = os.path.join(txt_dir, "test.ru")

    print("Загрузка Multilingual TEDx Russian (локальный датасет)...")

    with open(yaml_path, encoding="utf-8") as f:
        segments = yaml.safe_load(f)          # list of dicts

    with open(ref_path, encoding="utf-8") as f:
        transcriptions = [line.rstrip("\n") for line in f]

    assert len(segments) == len(transcriptions), (
        f"Число сегментов ({len(segments)}) ≠ число транскрипций ({len(transcriptions)})"
    )

    if max_samples:
        segments       = segments[:max_samples]
        transcriptions = transcriptions[:max_samples]

    # Кэш уже загруженных FLAC-файлов (по имени файла)
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    samples = []
    for seg, ref in zip(segments, transcriptions):
        wav_name = seg["wav"]                        # напр. "6hoeGxsHS6c.wav"
        # Файл может оказаться .flac
        wav_path = os.path.join(wav_dir, wav_name)
        if not os.path.exists(wav_path):
            stem = os.path.splitext(wav_name)[0]
            for ext in (".flac", ".wav", ".mp3"):
                candidate = os.path.join(wav_dir, stem + ext)
                if os.path.exists(candidate):
                    wav_path = candidate
                    break

        if wav_path not in audio_cache:
            arr, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            audio_cache[wav_path] = (arr, sr)

        arr, sr = audio_cache[wav_path]

        start = int(seg["offset"] * sr)
        end   = start + int(seg["duration"] * sr)
        chunk = arr[start:end]

        # Ресэмплинг до 16 000 Гц при необходимости
        if sr != 16000:
            import torchaudio.transforms as T
            chunk = T.Resample(sr, 16000)(
                torch.tensor(chunk)
            ).numpy()

        samples.append({"audio": chunk.astype(np.float32), "transcription": ref})

    print(f"  Загружено: {len(samples)} сегментов")
    return samples


# ══════════════════════════════════════════════════════════════
#  Инференс — Whisper baseline / FT
# ══════════════════════════════════════════════════════════════

def _whisper_infer(samples, use_lora: bool, cfg: Config) -> list[str]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    print(f"  Загрузка Whisper ({'FT + ПО' if use_lora else 'baseline'})...")
    base = WhisperForConditionalGeneration.from_pretrained(
        cfg.MODEL_NAME, torch_dtype=dtype
    ).to(device).eval()

    if use_lora and Path(cfg.OUTPUT_DIR).exists():
        model = PeftModel.from_pretrained(base, cfg.OUTPUT_DIR).eval()
    else:
        model = base

    processor = WhisperProcessor.from_pretrained(
        cfg.MODEL_NAME, language=cfg.LANGUAGE, task=cfg.TASK
    )

    preds = []
    for i, sample in enumerate(samples, 1):
        arr  = sample["audio"]
        feats = processor(arr, sampling_rate=16000,
                          return_tensors="pt").input_features.to(device, dtype=dtype)
        with torch.inference_mode():
            ids = model.generate(feats, language=cfg.LANGUAGE,
                                 task=cfg.TASK, max_new_tokens=225)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if use_lora:
            try:
                pp   = postprocess.process(text, restore_punct=False,
                                           normalize_nums=False, run_ner=False)
                text = pp.text_final
            except Exception:
                pass
        preds.append(text)
        if i % 50 == 0:
            print(f"    {i}/{len(samples)}")

    del model
    torch.cuda.empty_cache()
    return preds


# ══════════════════════════════════════════════════════════════
#  Инференс — GigaAM v3
# ══════════════════════════════════════════════════════════════

def _gigaam_infer(samples) -> list[str] | None:
    try:
        import gigaam
    except ImportError:
        print("  [пропуск] gigaam не установлен")
        return None

    GIGAAM_MAX = 24 * 16000  # 24 секунды — чуть меньше лимита 25 с

    def _transcribe_chunk(m, arr16k):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, arr16k, 16000, subtype="PCM_16")
        tmp.close()
        result = m.transcribe(tmp.name)
        os.unlink(tmp.name)
        t = result.text if hasattr(result, "text") else str(result)
        return t.strip()

    print("  Загрузка GigaAM v3...")
    model = gigaam.load_model("v3_e2e_rnnt")
    preds = []
    for i, sample in enumerate(samples, 1):
        arr = sample["audio"]
        if len(arr) <= GIGAAM_MAX:
            text = _transcribe_chunk(model, arr)
        else:
            # Чанкинг: шаг 20 с, перекрытие 4 с
            step, overlap = 20 * 16000, 4 * 16000
            parts = []
            pos = 0
            while pos < len(arr):
                chunk = arr[pos: pos + GIGAAM_MAX]
                parts.append(_transcribe_chunk(model, chunk))
                pos += step
            text = " ".join(p for p in parts if p)
        preds.append(text)
        if i % 50 == 0:
            print(f"    {i}/{len(samples)}")
    return preds


# ══════════════════════════════════════════════════════════════
#  Инференс — Vikhr Borealis
# ══════════════════════════════════════════════════════════════

def _borealis_infer(samples) -> list[str]:
    from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("  Загрузка Vikhr Borealis...")
    extractor = AutoFeatureExtractor.from_pretrained(
        "Vikhrmodels/Borealis", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "Vikhrmodels/Borealis", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Vikhrmodels/Borealis", trust_remote_code=True, torch_dtype=torch.float16
    ).to(device).eval()

    preds = []
    for i, sample in enumerate(samples, 1):
        arr  = sample["audio"]
        proc = extractor(arr, sampling_rate=16000, padding="max_length",
                         max_length=480_000, return_attention_mask=True,
                         return_tensors="pt")
        mel  = proc.input_features.squeeze(0).to(device)
        att  = proc.attention_mask.squeeze(0).to(device)
        with torch.inference_mode():
            out = model.generate(mel=mel, att_mask=att,
                                 max_new_tokens=350, do_sample=False)
        if isinstance(out, str):
            text = out
        elif isinstance(out, list) and out and isinstance(out[0], str):
            text = out[0]
        else:
            text = tokenizer.decode(out[0], skip_special_tokens=True)
        preds.append(text.strip())
        if i % 50 == 0:
            print(f"    {i}/{len(samples)}")

    del model
    torch.cuda.empty_cache()
    return preds


# ══════════════════════════════════════════════════════════════
#  Сохранение / загрузка предсказаний
# ══════════════════════════════════════════════════════════════

def save_predictions(refs: list[str], preds: list[str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r, p in zip(refs, preds):
            f.write(f"REF: {r}\nPRED: {p}\n\n")
    print(f"  Сохранено: {path} ({len(refs)} примеров)")


def load_predictions(path: str):
    refs, preds = [], []
    with open(path, encoding="utf-8") as f:
        content = f.read()
    for block in content.strip().split("\n\n"):
        lines = block.strip().splitlines()
        if len(lines) >= 2:
            refs.append(lines[0][5:])
            preds.append(lines[1][6:])
    return refs, preds


# ══════════════════════════════════════════════════════════════
#  Графики
# ══════════════════════════════════════════════════════════════

COLORS = {
    "Whisper baseline": "#4C72B0",
    "GigaAM v3":        "#55A868",
    "Vikhr Borealis":   "#C44E52",
    "Whisper FT + ПО":  "#DD8452",
}


def plot_results(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    models = list(results.keys())
    colors = [COLORS.get(m, "#888") for m in models]
    wers   = [results[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(models)), wers, color=colors, alpha=0.85,
                  edgecolor="white", width=0.5)
    ymax = max(wers)
    for bar, v in zip(bars, wers):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.018,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("WER без пунктуации, %", fontsize=12)
    ax.set_title("WER без пунктуации на Multilingual TEDx Russian ↓",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, ymax * 1.22)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(out_dir, "tedx_wer.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранён: {path}")


# ══════════════════════════════════════════════════════════════
#  Точка входа
# ══════════════════════════════════════════════════════════════

MODEL_CHOICES = ["baseline", "ft", "gigaam", "borealis"]

PRED_PATHS = {
    "baseline": "predictions_tedx_baseline.txt",
    "ft":       "predictions_tedx_ft.txt",
    "gigaam":   "predictions_tedx_gigaam.txt",
    "borealis": "predictions_tedx_borealis.txt",
}

DISPLAY_NAMES = {
    "baseline": "Whisper baseline",
    "gigaam":   "GigaAM v3",
    "borealis": "Vikhr Borealis",
    "ft":       "Whisper FT + ПО",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES,
                        default=MODEL_CHOICES,
                        help="Какие модели запускать (default: все)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Ограничить число примеров (для отладки)")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()
    cfg  = Config()

    # ── Загрузка датасета ─────────────────────────────────────
    samples = load_tedx(args.max_samples)
    refs    = [s["transcription"] for s in samples]
    print(f"  Примеры транскрипций (первые 3):")
    for r in refs[:3]:
        print(f"    {r}")

    # ── Инференс по выбранным моделям ─────────────────────────
    for model_key in args.models:
        pred_path = PRED_PATHS[model_key]

        if os.path.exists(pred_path):
            print(f"\n[{DISPLAY_NAMES[model_key]}] уже есть → {pred_path}, пропуск")
            continue

        print(f"\n[ASR] {DISPLAY_NAMES[model_key]}")
        t0 = time.time()

        if model_key == "baseline":
            preds = _whisper_infer(samples, use_lora=False, cfg=cfg)
        elif model_key == "ft":
            preds = _whisper_infer(samples, use_lora=True, cfg=cfg)
        elif model_key == "gigaam":
            preds = _gigaam_infer(samples)
        elif model_key == "borealis":
            preds = _borealis_infer(samples)

        if preds is None:
            print(f"  [пропуск] {DISPLAY_NAMES[model_key]}")
            continue

        elapsed = time.time() - t0
        print(f"  Время инференса: {elapsed:.0f} с ({elapsed/len(samples):.2f} с/пример)")
        save_predictions(refs, preds, pred_path)

    # ── Вычисление WER по всем сохранённым файлам ─────────────
    print("\n" + "=" * 65)
    print("WER БЕЗ ПУНКТУАЦИИ на Multilingual TEDx Russian")
    print("  Нормализация: нижний регистр + удаление пунктуации")
    print("-" * 65)

    results_wer = {}
    rows = []
    for model_key in MODEL_CHOICES:
        pred_path = PRED_PATHS[model_key]
        if not os.path.exists(pred_path):
            continue
        r, p = load_predictions(pred_path)
        wer  = compute_wer(r, p)
        name = DISPLAY_NAMES[model_key]
        results_wer[name] = wer
        rows.append({"system": name, "wer_norm": round(wer, 4), "n": len(r)})
        print(f"  {name:<22}  WER_n = {wer:.2f}%  (n={len(r)})")

    print("=" * 65)

    if not results_wer:
        print("Нет результатов — запусти скрипт с нужными моделями.")
        return

    # ── Относительное улучшение vs baseline ───────────────────
    if "Whisper baseline" in results_wer:
        base = results_wer["Whisper baseline"]
        print("\nОтносительное улучшение WER_n vs Whisper baseline:")
        for name, wer in results_wer.items():
            if name == "Whisper baseline":
                continue
            rel  = (base - wer) / base * 100
            sign = "улучшение" if rel > 0 else "ухудшение"
            print(f"  {name}: {rel:+.1f}% ({sign})")

    # ── Сохранение CSV и графика ──────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "tedx_comparison.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nCSV сохранён: {csv_path}")

    print("Построение графика...")
    plot_results(results_wer, out_dir=args.out_dir)
    print(f"\nГотово. Файлы:")
    print(f"  {csv_path}")
    print(f"  {os.path.join(args.out_dir, 'tedx_wer.png')}")


if __name__ == "__main__":
    main()
