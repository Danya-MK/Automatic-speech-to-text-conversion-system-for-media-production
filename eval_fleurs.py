"""
Оценка ASR-систем на Multilingual LibriSpeech Russian
(facebook/multilingual_librispeech, config "russian").

MLS Russian содержит транскрипции из аудиокниг Project Gutenberg —
пунктуация сохранена из исходного текста, что позволяет корректно
измерить Punct F1 (в отличие от Common Voice).

Метрики:
  WER_n   — WER без пунктуации (чистая лексика)
  WER     — WER с пунктуацией
  CER     — Character Error Rate
  PunctF1 — F1 по знакам . , ! ? ; :
  RTF     — Real-Time Factor (время инференса / длина аудио)

Использование:
    python eval_fleurs.py                          # все 3 модели
    python eval_fleurs.py --models baseline ft     # без GigaAM
    python eval_fleurs.py --max_samples 50         # быстрый тест
"""
import argparse
import os
import re
import tempfile
import time
from pathlib import Path

import evaluate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio

matplotlib.rcParams["font.family"] = "DejaVu Sans"

import postprocess
from config import Config

# ══════════════════════════════════════════════════════════════
#  Метрики
# ══════════════════════════════════════════════════════════════

_wer_metric = evaluate.load("wer")
_cer_metric = evaluate.load("cer")
PUNCT_RE = re.compile(r"[,\.!?;:]")


def _normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def compute_metrics(refs: list, preds: list) -> dict:
    refs_n  = [_normalize(r) for r in refs]
    preds_n = [_normalize(p) for p in preds]

    wer_p = 100 * _wer_metric.compute(predictions=preds,   references=refs)
    wer_n = 100 * _wer_metric.compute(predictions=preds_n, references=refs_n)
    cer_v = 100 * _cer_metric.compute(predictions=preds,   references=refs)

    tp = fp = fn = 0
    for r, p in zip(refs, preds):
        rm  = PUNCT_RE.findall(r)
        pm  = PUNCT_RE.findall(p)
        _tp = sum(1 for m in pm if m in rm)
        tp += _tp
        fp += len(pm) - _tp
        fn += len(rm) - _tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pf1  = 100 * (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {"wer_punct": wer_p, "wer_norm": wer_n, "cer": cer_v, "punct_f1": pf1}


# ══════════════════════════════════════════════════════════════
#  Загрузка Multilingual LibriSpeech Russian
# ══════════════════════════════════════════════════════════════

def load_fleurs(max_samples: int = None):
    """
    Загружает тестовый сплит MLS Russian (facebook/multilingual_librispeech).
    Аудио в FLAC 16 kHz, datasets декодирует через soundfile (без torchcodec).
    Транскрипции из аудиокниг Project Gutenberg — сохранена пунктуация.
    Возвращает (audio_arrays, refs): массивы float32 16 kHz и тексты с пунктуацией.
    """
    from datasets import load_dataset

    print("Загрузка Multilingual LibriSpeech Russian (test)...")
    ds = load_dataset(
        "facebook/multilingual_librispeech",
        "russian",
        split="test",
        trust_remote_code=False,
    )

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    audio_arrays, refs = [], []
    for ex in ds:
        # поле "transcript" — текст из аудиокниги с пунктуацией
        refs.append(ex["transcript"])
        audio_arrays.append(ex["audio"]["array"].astype("float32"))

    print(f"  Загружено: {len(refs)} примеров, "
          f"~{sum(len(a)/16000 for a in audio_arrays)/60:.1f} мин аудио")
    return audio_arrays, refs


# ══════════════════════════════════════════════════════════════
#  Инференс: Whisper (baseline / fine-tuned)
# ══════════════════════════════════════════════════════════════

def run_whisper(audio_arrays: list, cfg, use_lora: bool = False):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    label = "Whisper FT" if use_lora else "Whisper baseline"
    print(f"  Загрузка {label}...")

    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    if use_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.OUTPUT_DIR).eval()

    processor = WhisperProcessor.from_pretrained(
        cfg.MODEL_NAME, language=cfg.LANGUAGE, task=cfg.TASK
    )

    dtype       = next(model.parameters()).dtype
    preds       = []
    t_infer     = 0.0
    total_audio = sum(len(a) / 16000 for a in audio_arrays)

    for i, arr in enumerate(audio_arrays, 1):
        inputs = processor(arr, sampling_rate=16000, return_tensors="pt")
        feats  = inputs.input_features.to(model.device, dtype=dtype)
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(
                feats,
                language=cfg.LANGUAGE,
                task=cfg.TASK,
                max_new_tokens=225,
            )
        t_infer += time.perf_counter() - t0
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        preds.append(text)
        if i % 50 == 0:
            print(f"    {i}/{len(audio_arrays)}")

    rtf = t_infer / total_audio if total_audio > 0 else float("nan")
    print(f"    RTF={rtf:.3f}  ({len(preds)} примеров)")

    del model
    torch.cuda.empty_cache()
    return preds, rtf


# ══════════════════════════════════════════════════════════════
#  Инференс: GigaAM v3
# ══════════════════════════════════════════════════════════════

def run_gigaam(audio_arrays: list):
    try:
        import gigaam
    except ImportError:
        print("  [пропуск] gigaam не установлен (pip install gigaam --no-deps)")
        return None, float("nan")

    print("  Загрузка GigaAM v3...")
    model = gigaam.load_model("v3_e2e_rnnt")

    preds       = []
    t_infer     = 0.0
    total_audio = sum(len(a) / 16000 for a in audio_arrays)

    for i, arr in enumerate(audio_arrays, 1):
        w   = torch.from_numpy(arr).unsqueeze(0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(tmp.name, w, 16000)
        tmp.close()

        t0 = time.perf_counter()
        text = model.transcribe(tmp.name)
        t_infer += time.perf_counter() - t0
        os.unlink(tmp.name)

        preds.append(text if isinstance(text, str) else str(text))
        if i % 50 == 0:
            print(f"    {i}/{len(audio_arrays)}")

    rtf = t_infer / total_audio if total_audio > 0 else float("nan")
    print(f"    RTF={rtf:.3f}  ({len(preds)} примеров)")
    return preds, rtf


# ══════════════════════════════════════════════════════════════
#  Вспомогательные
# ══════════════════════════════════════════════════════════════

def save_predictions(refs: list, preds: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r, p in zip(refs, preds):
            f.write(f"REF:  {r}\nPRED: {p}\n\n")
    print(f"  Предсказания: {path}")


COLORS = {
    "Whisper baseline": "#4C72B0",
    "GigaAM v3":        "#55A868",
    "Whisper FT + PP":  "#DD8452",
}


# ══════════════════════════════════════════════════════════════
#  Графики
# ══════════════════════════════════════════════════════════════

def plot_comparison(metrics: dict, rtf_vals: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    models = list(metrics.keys())
    colors = [COLORS.get(m, "#888888") for m in models]

    def bar_ax(ax, values, ylabel, title, lower_better=True):
        bars = ax.bar(models, values, color=colors, alpha=0.85,
                      edgecolor="white", width=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.015,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("↓ лучше" if lower_better else "↑ лучше",
                      fontsize=8, color="gray")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Сравнение ASR-систем на MLS Russian (test)\n"
        "Транскрипции из аудиокниг — корректная оценка пунктуации",
        fontsize=13, fontweight="bold",
    )

    bar_ax(axes[0, 0],
           [metrics[m]["wer_norm"]  for m in models],
           "WER, %", "WER без пунктуации (лексика) ↓")
    bar_ax(axes[0, 1],
           [metrics[m]["wer_punct"] for m in models],
           "WER, %", "WER с пунктуацией ↓")
    bar_ax(axes[1, 0],
           [metrics[m]["cer"]       for m in models],
           "CER, %", "Character Error Rate ↓")
    bar_ax(axes[1, 1],
           [metrics[m]["punct_f1"]  for m in models],
           "F1, %", "Punctuation F1 ↑", lower_better=False)

    plt.tight_layout()
    path = os.path.join(out_dir, "fleurs_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  График: {path}")

    # RTF — горизонтальный bar
    valid = {m: rtf_vals[m] for m in models
             if not np.isnan(rtf_vals.get(m, float("nan")))}
    if valid:
        ms  = list(valid.keys())
        vs  = list(valid.values())
        cs  = [COLORS.get(m, "#888") for m in ms]
        fig2, ax2 = plt.subplots(figsize=(8, max(3, len(ms) * 1.2)))
        bars = ax2.barh(ms, vs, color=cs, alpha=0.85, edgecolor="white")
        ax2.axvline(1.0, color="red", linestyle="--", linewidth=1.2,
                    label="RTF = 1 (реальное время)")
        for bar, v in zip(bars, vs):
            ax2.text(v + max(vs) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{v:.3f}", va="center", fontsize=10)
        ax2.set_xlabel("RTF  (↓ лучше)", fontsize=10)
        ax2.set_title("Real-Time Factor на MLS Russian", fontsize=11,
                      fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(axis="x", alpha=0.3)
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path2 = os.path.join(out_dir, "fleurs_rtf.png")
        plt.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  График RTF: {path2}")


# ══════════════════════════════════════════════════════════════
#  Точка входа
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Оценка ASR на FLEURS Russian"
    )
    parser.add_argument("--out_dir",      default="results",
                        help="Папка для результатов")
    parser.add_argument("--max_samples",  type=int, default=None,
                        help="Ограничить число примеров (для быстрого теста)")
    parser.add_argument("--models",       nargs="+",
                        choices=["baseline", "ft", "gigaam"],
                        default=["baseline", "ft", "gigaam"],
                        help="Какие модели запускать")
    args = parser.parse_args()
    cfg = Config()

    # ── Загрузка данных ───────────────────────────────────────
    audio_arrays, refs = load_fleurs(args.max_samples)

    all_preds: dict[str, list] = {}
    all_rtf:   dict[str, float] = {}

    # ── Инференс ──────────────────────────────────────────────
    if "baseline" in args.models:
        print("\n[1/3] Whisper baseline")
        preds, rtf = run_whisper(audio_arrays, cfg, use_lora=False)
        all_preds["Whisper baseline"] = preds
        all_rtf["Whisper baseline"]   = rtf
        save_predictions(refs, preds, "predictions_fleurs_baseline.txt")

    if "ft" in args.models:
        if not Path(cfg.OUTPUT_DIR).exists():
            print(f"\n[2/3] Whisper FT: [пропуск] адаптеры не найдены: {cfg.OUTPUT_DIR}")
        else:
            print("\n[2/3] Whisper FT + постобработка")
            preds, rtf = run_whisper(audio_arrays, cfg, use_lora=True)

            print("  Применение постобработки...")
            pp_preds = []
            for p in preds:
                try:
                    pp_preds.append(
                        postprocess.process(p, run_ner=False).text_final
                    )
                except Exception:
                    pp_preds.append(p)

            all_preds["Whisper FT + PP"] = pp_preds
            all_rtf["Whisper FT + PP"]   = rtf
            save_predictions(refs, pp_preds, "predictions_fleurs_ft.txt")

    if "gigaam" in args.models:
        print("\n[3/3] GigaAM v3")
        preds, rtf = run_gigaam(audio_arrays)
        if preds is not None:
            all_preds["GigaAM v3"] = preds
            all_rtf["GigaAM v3"]   = rtf
            save_predictions(refs, preds, "predictions_fleurs_gigaam.txt")

    if not all_preds:
        print("\nНет предсказаний — проверь аргументы --models.")
        return

    # ── Метрики ───────────────────────────────────────────────
    print("\nВычисление метрик...")
    metrics: dict[str, dict] = {}
    for name, preds in all_preds.items():
        n   = len(preds)
        ref = refs[:n]
        m   = compute_metrics(ref, preds)
        metrics[name] = m

    # ── Таблица ───────────────────────────────────────────────
    W = 85
    print("\n" + "=" * W)
    print("MLS Russian — результаты (транскрипции из аудиокниг с пунктуацией)")
    print("-" * W)
    print(f"  {'Система':<22} {'WER':>7} {'WER_n':>7} {'CER':>7} {'PunctF1':>9} {'RTF':>8}")
    print("-" * W)

    baseline_wer = metrics.get("Whisper baseline", {}).get("wer_norm")
    for name, m in metrics.items():
        rtf_s = (f"{all_rtf[name]:.3f}"
                 if not np.isnan(all_rtf.get(name, float("nan"))) else "  -")
        delta = ""
        if baseline_wer and name != "Whisper baseline":
            d     = m["wer_norm"] - baseline_wer
            delta = f"  ({d:+.1f}%)"
        print(f"  {name:<22} {m['wer_punct']:>6.2f}%  {m['wer_norm']:>6.2f}%  "
              f"{m['cer']:>6.2f}%  {m['punct_f1']:>8.1f}%  {rtf_s:>7}{delta}")

    print("=" * W)

    if baseline_wer:
        print("\nОтносительное улучшение WER_n vs Whisper baseline:")
        for name, m in metrics.items():
            if name == "Whisper baseline":
                continue
            rel = (baseline_wer - m["wer_norm"]) / baseline_wer * 100
            sign = "улучшение" if rel > 0 else "ухудшение"
            print(f"  {name}: {rel:+.1f}% ({sign})")

    # ── Сохранение ────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    rows = [{"system": n, **m, "rtf": all_rtf.get(n, float("nan"))}
            for n, m in metrics.items()]
    csv_path = os.path.join(args.out_dir, "fleurs_comparison.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nCSV: {csv_path}")

    print("Построение графиков...")
    plot_comparison(metrics, all_rtf, args.out_dir)

    print(f"\nГотово. Результаты: {args.out_dir}/")


if __name__ == "__main__":
    main()
