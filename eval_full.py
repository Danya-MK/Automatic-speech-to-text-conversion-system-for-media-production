"""Полная экспериментальная оценка системы.

Считает: WER, CER, Punctuation Accuracy, RTF, NER F1
до и после постобработки. Строит таблицы и графики.

Использование:
    python eval_full.py --max_samples 500
"""
import argparse
import os
import re
import time
from pathlib import Path

import evaluate
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from peft import PeftModel
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import postprocess
from config import Config
from data_utils import find_cv_language_root


# ══════════════════════════════════════════════════════════════
#  Загрузка данных
# ══════════════════════════════════════════════════════════════

def load_test_df(cfg, max_samples):
    cv_root = find_cv_language_root(cfg.LOCAL_DATASET_PATH)
    clips   = os.path.join(cv_root, "clips")
    tsv     = os.path.join(cv_root, "test.tsv")
    df = pd.read_csv(tsv, sep="\t", low_memory=False)[["path", "sentence"]].dropna()
    df["path"] = df["path"].apply(lambda p: os.path.join(clips, p))
    return df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  Метрики
# ══════════════════════════════════════════════════════════════

_wer_metric = evaluate.load("wer")
_cer_metric = evaluate.load("cer")


def compute_wer(refs, preds):
    return 100 * _wer_metric.compute(predictions=preds, references=refs)


def compute_cer(refs, preds):
    return 100 * _cer_metric.compute(predictions=preds, references=refs)


PUNCT_RE = re.compile(r"[,\.!?;:—\-–«»]")


def punctuation_accuracy(refs: list[str], preds: list[str]) -> float:
    """
    Доля совпадающих знаков пунктуации (по позициям в тексте).
    Простая метрика: F1 между множествами знаков в референсе и предсказании.
    """
    total_tp = total_fp = total_fn = 0
    for ref, pred in zip(refs, preds):
        ref_marks  = PUNCT_RE.findall(ref)
        pred_marks = PUNCT_RE.findall(pred)
        tp = sum(1 for p in pred_marks if p in ref_marks)
        fp = len(pred_marks) - tp
        fn = len(ref_marks)  - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    if total_tp + total_fp == 0:
        return 0.0
    precision = total_tp / (total_tp + total_fp)
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)  # F1


def normalize_for_wer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


# ══════════════════════════════════════════════════════════════
#  Инференс с измерением RTF
# ══════════════════════════════════════════════════════════════

def transcribe_batch(paths, model, processor, cfg, target_sr=16000):
    """Транскрибирует батч файлов, возвращает (тексты, суммарная_длительность_аудио, время_инференса)."""
    audio_arrays, total_audio_sec = [], 0.0

    for path in paths:
        try:
            waveform, sr = torchaudio.load(path)
            if sr != target_sr:
                waveform = T.Resample(sr, target_sr)(waveform)
            arr = waveform.mean(0).numpy()
            audio_arrays.append(arr)
            total_audio_sec += len(arr) / target_sr
        except Exception:
            audio_arrays.append(None)

    valid = [(i, a) for i, a in enumerate(audio_arrays) if a is not None]
    if not valid:
        return [], 0.0, 0.0

    idxs, arrs = zip(*valid)
    inputs = processor(list(arrs), sampling_rate=target_sr,
                       return_tensors="pt", padding=True)
    dtype  = next(model.parameters()).dtype
    feats  = inputs.input_features.to(device=model.device, dtype=dtype)

    t0 = time.perf_counter()
    with torch.no_grad():
        ids = model.generate(feats, language=cfg.LANGUAGE, task=cfg.TASK,
                             max_new_tokens=225)
    elapsed = time.perf_counter() - t0

    texts_raw = processor.batch_decode(ids, skip_special_tokens=True)
    texts = [""] * len(paths)
    for i, txt in zip(idxs, texts_raw):
        texts[i] = txt.lower().strip()

    return texts, total_audio_sec, elapsed


# ══════════════════════════════════════════════════════════════
#  Главный цикл оценки
# ══════════════════════════════════════════════════════════════

def evaluate_model(df, model, processor, cfg, batch_size, desc, run_postprocess):
    predictions_raw, predictions_norm, predictions_pp = [], [], []
    references = []
    total_audio_sec = total_infer_sec = 0.0

    for i in tqdm(range(0, len(df), batch_size), desc=desc):
        batch   = df.iloc[i: i + batch_size]
        paths   = batch["path"].tolist()
        refs    = [s.lower().strip() for s in batch["sentence"].tolist()]

        texts, audio_dur, infer_dur = transcribe_batch(
            paths, model, processor, cfg
        )
        total_audio_sec += audio_dur
        total_infer_sec += infer_dur

        for txt, ref in zip(texts, refs):
            if txt:
                predictions_raw.append(txt)
                predictions_norm.append(normalize_for_wer(txt))
                references.append(ref)

                if run_postprocess:
                    pp = postprocess.process(txt, restore_punct=True,
                                             normalize_nums=True, run_ner=True)
                    predictions_pp.append(pp.text_final.lower().strip())
                else:
                    predictions_pp.append(txt)

    rtf = total_infer_sec / total_audio_sec if total_audio_sec > 0 else float("inf")

    refs_norm = [normalize_for_wer(r) for r in references]
    pp_norm   = [normalize_for_wer(p) for p in predictions_pp]

    return {
        "n":             len(references),
        "rtf":           rtf,
        # С пунктуацией (ref с пункт. vs pred с пункт.)
        "wer_raw":       compute_wer(references,   predictions_raw),
        "cer_raw":       compute_cer(references,   predictions_raw),
        # Без пунктуации (ref без пункт. vs pred без пункт.)
        "wer_norm":      compute_wer(refs_norm,    predictions_norm),
        "cer_norm":      compute_cer(refs_norm,    predictions_norm),
        # После постобработки (ref с пункт. vs pred+PP с пункт.)
        "wer_pp":        compute_wer(references,   predictions_pp),
        "cer_pp":        compute_cer(references,   predictions_pp),
        # После постобработки без пунктуации
        "wer_pp_norm":   compute_wer(refs_norm,    pp_norm),
        "cer_pp_norm":   compute_cer(refs_norm,    pp_norm),
        # Punctuation F1
        "punct_f1_raw":  punctuation_accuracy(references, predictions_raw),
        "punct_f1_pp":   punctuation_accuracy(references, predictions_pp),
        # Сырые тексты для сохранения
        "refs":          references,
        "refs_norm":     refs_norm,
        "preds_raw":     predictions_raw,
        "preds_norm":    predictions_norm,
        "preds_pp":      predictions_pp,
        "preds_pp_norm": pp_norm,
    }


# ══════════════════════════════════════════════════════════════
#  Графики
# ══════════════════════════════════════════════════════════════

def plot_results(results: dict, out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)

    models  = list(results.keys())
    wer_raw = [results[m]["wer_raw"]  for m in models]
    wer_pp  = [results[m]["wer_pp"]   for m in models]
    cer_raw = [results[m]["cer_raw"]  for m in models]
    cer_pp  = [results[m]["cer_pp"]   for m in models]
    punct   = [results[m]["punct_f1_raw"] * 100 for m in models]
    punct_p = [results[m]["punct_f1_pp"]  * 100 for m in models]
    rtf     = [results[m]["rtf"]     for m in models]

    x = range(len(models))
    w = 0.35

    # ── WER до/после постобработки ──
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w/2 for i in x], wer_raw, w, label="До постобработки",  color="#4C72B0")
    ax.bar([i + w/2 for i in x], wer_pp,  w, label="После постобработки", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("WER, %")
    ax.set_title("Word Error Rate (WER)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    for i, (a, b) in enumerate(zip(wer_raw, wer_pp)):
        ax.text(i - w/2, a + 0.2, f"{a:.1f}%", ha="center", fontsize=8)
        ax.text(i + w/2, b + 0.2, f"{b:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wer_comparison.png"), dpi=150)
    plt.close()

    # ── CER ──
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w/2 for i in x], cer_raw, w, label="До постобработки",  color="#4C72B0")
    ax.bar([i + w/2 for i in x], cer_pp,  w, label="После постобработки", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("CER, %")
    ax.set_title("Character Error Rate (CER)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cer_comparison.png"), dpi=150)
    plt.close()

    # ── Punctuation F1 ──
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w/2 for i in x], punct,   w, label="До постобработки",  color="#4C72B0")
    ax.bar([i + w/2 for i in x], punct_p, w, label="После постобработки", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Punctuation F1, %")
    ax.set_title("Punctuation Accuracy (F1)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "punct_accuracy.png"), dpi=150)
    plt.close()

    # ── RTF ──
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(models, rtf, color="#55A868")
    ax.axvline(1.0, color="red", linestyle="--", label="RTF = 1 (реальное время)")
    ax.set_xlabel("RTF (Real-Time Factor)")
    ax.set_title("Скорость обработки (меньше = быстрее)")
    ax.legend()
    for bar, val in zip(bars, rtf):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rtf.png"), dpi=150)
    plt.close()

    print(f"Графики сохранены в '{out_dir}/'")


# ══════════════════════════════════════════════════════════════
#  Точка входа
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples",  type=int, default=300)
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--out_dir",      type=str, default="results")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Пропустить оценку baseline (экономит время)")
    args = parser.parse_args()

    cfg = Config()
    df  = load_test_df(cfg, args.max_samples)
    print(f"Тестовых примеров: {len(df)}")

    results = {}

    # ── Базовая модель ──────────────────────────────────────
    if not args.skip_baseline:
        print("\n[1/2] Базовая модель Whisper-large-v3…")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            cfg.MODEL_NAME, dtype=torch.float16, device_map="auto"
        )
        base_model.eval()
        processor = WhisperProcessor.from_pretrained(
            cfg.MODEL_NAME, language=cfg.LANGUAGE, task=cfg.TASK
        )
        results["Whisper baseline"] = evaluate_model(
            df, base_model, processor, cfg,
            args.batch_size, "Baseline", run_postprocess=True
        )
        del base_model
        torch.cuda.empty_cache()

    # ── Дообученная модель ──────────────────────────────────
    print("\n[2/2] Дообученная модель (Whisper + LoRA)…")
    base2 = WhisperForConditionalGeneration.from_pretrained(
        cfg.MODEL_NAME, dtype=torch.float16, device_map="auto"
    )
    ft_model = PeftModel.from_pretrained(base2, cfg.OUTPUT_DIR)
    ft_model.eval()
    processor2 = WhisperProcessor.from_pretrained(
        cfg.MODEL_NAME, language=cfg.LANGUAGE, task=cfg.TASK
    )
    results["Whisper + LoRA"] = evaluate_model(
        df, ft_model, processor2, cfg,
        args.batch_size, "Fine-tuned", run_postprocess=True
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Итоговая таблица ────────────────────────────────────
    W = 95
    print("\n" + "═" * W)
    print("Сравниваемые метрики:")
    print("  WER_punct  — ref с пункт.  vs pred с пункт.  (Whisper-выход)")
    print("  WER_norm   — ref без пункт. vs pred без пункт. (чистая лексика)")
    print("  WER_pp     — ref с пункт.  vs pred после постобработки")
    print("  WER_pp_n   — ref без пункт. vs pred+PP без пункт.")
    print("─" * W)
    hdr = (f"{'Модель':<22} {'WER_punct':>9} {'WER_norm':>9} "
           f"{'WER_pp':>8} {'WER_pp_n':>9} {'CER':>6} {'PunctF1':>8} {'RTF':>7}")
    print(hdr)
    print("─" * W)
    for name, r in results.items():
        print(
            f"{name:<22} "
            f"{r['wer_raw']:>8.2f}%  "
            f"{r['wer_norm']:>7.2f}%  "
            f"{r['wer_pp']:>7.2f}%  "
            f"{r['wer_pp_norm']:>7.2f}%  "
            f"{r['cer_raw']:>5.2f}%  "
            f"{r['punct_f1_raw']*100:>7.1f}%  "
            f"{r['rtf']:>6.3f}"
        )
    print("═" * W)

    # ── Детальный CSV с транскрипциями ──────────────────────
    for name, r in results.items():
        safe_name = name.replace(" ", "_").replace("+", "plus")
        detail_path = os.path.join(args.out_dir, f"transcriptions_{safe_name}.csv")
        pd.DataFrame({
            "reference":          r["refs"],
            "reference_no_punct": r["refs_norm"],
            "asr_with_punct":     r["preds_raw"],
            "asr_no_punct":       r["preds_norm"],
            "asr_postprocessed":  r["preds_pp"],
            "asr_pp_no_punct":    r["preds_pp_norm"],
        }).to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"Транскрипции сохранены: {detail_path}")

    # ── Сводный CSV метрик ──────────────────────────────────
    rows = []
    for name, r in results.items():
        rows.append({
            "model":        name,
            "n":            r["n"],
            "wer_punct":    round(r["wer_raw"],      2),
            "wer_norm":     round(r["wer_norm"],     2),
            "wer_pp":       round(r["wer_pp"],       2),
            "wer_pp_norm":  round(r["wer_pp_norm"],  2),
            "cer":          round(r["cer_raw"],      2),
            "cer_pp":       round(r["cer_pp"],       2),
            "punct_f1_raw": round(r["punct_f1_raw"], 4),
            "punct_f1_pp":  round(r["punct_f1_pp"],  4),
            "rtf":          round(r["rtf"],           4),
        })
    metrics_path = os.path.join(args.out_dir, "results.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f"Метрики сохранены:     {metrics_path}")

    # ── Графики ─────────────────────────────────────────────
    plot_results(results, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
