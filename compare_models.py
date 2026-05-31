"""
Сравнительный анализ трёх систем распознавания речи:
  1. Whisper-large-v3 (baseline, без дообучения)
  2. GigaAM v3 e2e-rnnt
  3. Whisper-large-v3 fine-tuned + постобработка (наша система)

Метрики
───────
WER  (Word Error Rate) — доля ошибочных слов.
     WER = (S + D + I) / N × 100%
     S — замены, D — удаления, I — вставки, N — слов в референсе.

CER  (Character Error Rate) — то же, но на уровне символов.

Punct F1 — F1-мера по знакам пунктуации (. , ! ? ; :).
     Precision = TP / (TP + FP), Recall = TP / (TP + FN)
     F1 = 2·P·R / (P + R)

NER F1 — F1-мера по именованным сущностям (Natasha NER, точное совпадение).
     Сущность засчитывается как TP, если её текст и тип совпадают с референсом.

RTF  (Real-Time Factor) — отношение времени обработки к длине аудио.
     RTF = T_inference / T_audio  (RTF < 1 → быстрее реального времени)

Использование:
    python compare_models.py
    python compare_models.py --rtf_samples 10   # более точный замер RTF
    python compare_models.py --skip_rtf         # пропустить RTF
"""
import argparse
import os
import re
import time
from pathlib import Path

import evaluate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T

matplotlib.rcParams["font.family"] = "DejaVu Sans"

import postprocess
from config import Config
from data_utils import find_cv_language_root

# ══════════════════════════════════════════════════════════════
#  Файлы с предсказаниями
# ══════════════════════════════════════════════════════════════

PRED_FILES = {
    "Whisper baseline":  "predictions_baseline.txt",
    "GigaAM v3":         "predictions_gigaam.txt",
    "Vikhr Borealis":    "predictions_borealis.txt",
    "Whisper FT + PP":   "predictions_finetuned.txt",
}

# JSON с RTF, сохранённый eval_borealis.py
RTF_JSON_FILES = {
    "Vikhr Borealis": "rtf_borealis.json",
}

COLORS = {
    "Whisper baseline":  "#4C72B0",
    "GigaAM v3":         "#55A868",
    "Vikhr Borealis":    "#C44E52",
    "Whisper FT + PP":   "#DD8452",
}


# ══════════════════════════════════════════════════════════════
#  Чтение файлов предсказаний
# ══════════════════════════════════════════════════════════════

def load_pred_file(path: str):
    refs, preds = [], []
    with open(path, encoding="utf-8") as f:
        content = f.read()
    for block in content.strip().split("\n\n"):
        lines = block.strip().splitlines()
        ref  = next((l[4:].strip() for l in lines if l.startswith("REF:")),  None)
        pred = next((l[5:].strip() for l in lines if l.startswith("PRED:")), None)
        if ref is not None and pred is not None:
            refs.append(ref)
            preds.append(pred)
    return refs, preds


# ══════════════════════════════════════════════════════════════
#  WER / CER / Punct F1
# ══════════════════════════════════════════════════════════════

_wer = evaluate.load("wer")
_cer = evaluate.load("cer")
PUNCT_RE = re.compile(r"[,\.!?;:—\-–«»]")


def wer(refs, preds):
    return 100 * _wer.compute(predictions=preds, references=refs)


def cer(refs, preds):
    return 100 * _cer.compute(predictions=preds, references=refs)


def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def punct_f1(refs: list, preds: list) -> float:
    tp = fp = fn = 0
    for r, p in zip(refs, preds):
        r_marks = PUNCT_RE.findall(r)
        p_marks = PUNCT_RE.findall(p)
        _tp = sum(1 for m in p_marks if m in r_marks)
        tp += _tp
        fp += len(p_marks) - _tp
        fn += len(r_marks) - _tp
    if tp + fp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


# ══════════════════════════════════════════════════════════════
#  NER F1 (Natasha, точное совпадение текст + тип)
# ══════════════════════════════════════════════════════════════

def _ner_entities(text: str) -> frozenset:
    """Возвращает frozenset пар (текст_нижний, тип) для всех сущностей."""
    try:
        from natasha import Doc
        c = postprocess._get_natasha()
        doc = Doc(text)
        doc.segment(c["segmenter"])
        doc.tag_morph(c["morph_tagger"])
        doc.tag_ner(c["ner_tagger"])
        return frozenset((s.text.lower().strip(), s.type) for s in doc.spans)
    except Exception:
        return frozenset()


def _ner_entities_batch(texts: list, label: str = "") -> list:
    """Запускает NER на списке текстов с прогресс-индикатором."""
    result = []
    n = len(texts)
    for i, t in enumerate(texts, 1):
        result.append(_ner_entities(t))
        if i % 500 == 0:
            print(f"    NER {label}: {i}/{n}")
    return result


def ner_f1_score(ref_ents: list, preds: list) -> float:
    """Entity-level NER F1 (exact match по тексту и типу сущности)."""
    tp = fp = fn = 0
    for r_ents, p in zip(ref_ents, preds):
        p_ents = _ner_entities(p)
        tp += len(r_ents & p_ents)
        fp += len(p_ents - r_ents)
        fn += len(r_ents - p_ents)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 100 * (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


# ══════════════════════════════════════════════════════════════
#  Сводная функция метрик
# ══════════════════════════════════════════════════════════════

def compute_all(refs, preds, ref_ents=None) -> dict:
    refs_n  = [normalize(r) for r in refs]
    preds_n = [normalize(p) for p in preds]
    result = {
        "wer_punct": wer(refs,   preds),
        "wer_norm":  wer(refs_n, preds_n),
        "cer":       cer(refs,   preds),
        "punct_f1":  punct_f1(refs, preds) * 100,
        "ner_f1":    ner_f1_score(ref_ents, preds) if ref_ents else float("nan"),
        "n":         len(refs),
    }
    return result


# ══════════════════════════════════════════════════════════════
#  RTF benchmark
# ══════════════════════════════════════════════════════════════

def _load_audio_sample(paths, target_sr=16000):
    """Загружает аудиофайлы; при ошибке torchcodec — fallback через ffmpeg."""
    import subprocess, tempfile, soundfile as sf
    arrays, total = [], 0.0
    for p in paths:
        arr = None
        # Попытка через torchaudio
        try:
            w, sr = torchaudio.load(p)
            if sr != target_sr:
                w = T.Resample(sr, target_sr)(w)
            arr = w.mean(0).numpy()
        except Exception:
            pass
        # Fallback: ffmpeg → WAV temp → soundfile
        if arr is None:
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                subprocess.run(
                    ["ffmpeg", "-y", "-i", p, "-ar", str(target_sr),
                     "-ac", "1", "-f", "wav", tmp.name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                )
                data, _ = sf.read(tmp.name, dtype="float32")
                arr = data if data.ndim == 1 else data.mean(axis=1)
                os.unlink(tmp.name)
            except Exception:
                pass
        if arr is not None:
            arrays.append(arr)
            total += len(arr) / target_sr
    return arrays, total


def rtf_whisper(paths, model, processor, cfg):
    arrays, audio_sec = _load_audio_sample(paths)
    if not arrays:
        return float("nan")
    dtype  = next(model.parameters()).dtype
    inputs = processor(arrays, sampling_rate=cfg.SAMPLING_RATE,
                       return_tensors="pt", padding=True)
    feats  = inputs.input_features.to(model.device, dtype=dtype)
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(feats, language=cfg.LANGUAGE, task=cfg.TASK,
                       max_new_tokens=225)
    return (time.perf_counter() - t0) / audio_sec


def rtf_gigaam(paths):
    try:
        import gigaam, tempfile
        model = gigaam.load_model("v3_e2e_rnnt")
        total_audio = total_infer = 0.0
        for p in paths:
            try:
                w, sr = torchaudio.load(p)
                if sr != 16000:
                    w = T.Resample(sr, 16000)(w)
                total_audio += w.shape[-1] / 16000
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                torchaudio.save(tmp.name, w.mean(0, keepdim=True), 16000)
                tmp.close()
                t0 = time.perf_counter()
                model.transcribe(tmp.name)
                total_infer += time.perf_counter() - t0
                os.unlink(tmp.name)
            except Exception:
                pass
        return total_infer / total_audio if total_audio > 0 else float("nan")
    except Exception:
        return float("nan")


def rtf_borealis(paths):
    """Замер RTF для Vikhr Borealis."""
    try:
        import json as _json
        # Читаем из JSON, сохранённого eval_borealis.py (точные данные по всем 2000 примерам)
        if os.path.exists("rtf_borealis.json"):
            with open("rtf_borealis.json") as f:
                return _json.load(f)["rtf"]

        from transformers import AutoFeatureExtractor, AutoModelForCausalLM
        import time as _time
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = AutoFeatureExtractor.from_pretrained("Vikhrmodels/Borealis", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Vikhrmodels/Borealis", trust_remote_code=True, torch_dtype=torch.float16
        ).to(device).eval()
        total_audio = total_infer = 0.0
        for p in paths:
            try:
                w, sr = torchaudio.load(p)
                if sr != 16000:
                    w = T.Resample(sr, 16000)(w)
                arr = w.mean(0).numpy()
                total_audio += len(arr) / 16000
                proc = extractor(arr, sampling_rate=16000, padding="max_length",
                                 max_length=480_000, return_attention_mask=True,
                                 return_tensors="pt")
                mel = proc.input_features.squeeze(0).to(device)
                att = proc.attention_mask.squeeze(0).to(device)
                t0 = _time.perf_counter()
                with torch.inference_mode():
                    model.generate(mel=mel, att_mask=att, max_new_tokens=350, do_sample=False)
                total_infer += _time.perf_counter() - t0
            except Exception:
                pass
        del model; torch.cuda.empty_cache()
        return total_infer / total_audio if total_audio > 0 else float("nan")
    except Exception:
        return float("nan")


def measure_rtf(cfg, n_samples: int = 5) -> dict:
    cv_root = find_cv_language_root(cfg.LOCAL_DATASET_PATH)
    df      = pd.read_csv(os.path.join(cv_root, "test.tsv"), sep="\t", low_memory=False)
    df      = df[["path", "sentence"]].dropna()
    clips   = os.path.join(cv_root, "clips")
    df["path"] = df["path"].apply(lambda p: os.path.join(clips, p))
    paths   = df.sample(n=min(n_samples, len(df)), random_state=0)["path"].tolist()

    rtf_vals = {}
    print(f"\n  RTF benchmark на {len(paths)} файлах...")

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print("    Whisper baseline...")
    bm   = WhisperForConditionalGeneration.from_pretrained(
        cfg.MODEL_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    proc = WhisperProcessor.from_pretrained(cfg.MODEL_NAME,
                                            language=cfg.LANGUAGE, task=cfg.TASK)
    rtf_vals["Whisper baseline"] = rtf_whisper(paths, bm, proc, cfg)
    del bm; torch.cuda.empty_cache()

    print("    Whisper FT + PP...")
    from peft import PeftModel
    base2 = WhisperForConditionalGeneration.from_pretrained(
        cfg.MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    ft    = PeftModel.from_pretrained(base2, cfg.OUTPUT_DIR).eval()
    proc2 = WhisperProcessor.from_pretrained(cfg.MODEL_NAME,
                                             language=cfg.LANGUAGE, task=cfg.TASK)
    rtf_vals["Whisper FT + PP"] = rtf_whisper(paths, ft, proc2, cfg)
    del ft; torch.cuda.empty_cache()

    print("    GigaAM v3...")
    rtf_vals["GigaAM v3"] = rtf_gigaam(paths)

    print("    Vikhr Borealis...")
    rtf_vals["Vikhr Borealis"] = rtf_borealis(paths)

    return rtf_vals


# ══════════════════════════════════════════════════════════════
#  Графики  (2 × 3)
# ══════════════════════════════════════════════════════════════

def _single_bar(ax, models, values, ylabel, title, colors,
                lower_better=True, fmt=".1f", unit="%"):
    """Один столбчатый график — один файл."""
    bars = ax.bar(range(len(models)), values, color=colors, alpha=0.85,
                  edgecolor="white", width=0.5)
    ymax = max((v for v in values if not np.isnan(v)), default=1)
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.018,
                    f"{v:{fmt}}{unit}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("↓ лучше" if lower_better else "↑ лучше",
                  fontsize=9, color="gray")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, ymax * 1.22)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_all(metrics: dict, rtf_vals: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    models = list(metrics.keys())
    colors = [COLORS.get(m, "#888") for m in models]

    def _save(fname):
        plt.tight_layout()
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Сохранён: {path}")

    # ── WER без пунктуации ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    _single_bar(ax, models,
                [metrics[m]["wer_norm"] for m in models],
                "WER, %", "WER без пунктуации ↓", colors)
    _save("wer_norm.png")

    # ── CER ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    _single_bar(ax, models,
                [metrics[m]["cer"] for m in models],
                "CER, %", "Character Error Rate ↓", colors)
    _save("cer.png")

    # ── Punct F1 ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    _single_bar(ax, models,
                [metrics[m]["punct_f1"] for m in models],
                "F1, %", "Punctuation F1 ↑", colors, lower_better=False)
    _save("punct_f1.png")

    # ── RTF (горизонтальный) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = [(m, rtf_vals.get(m, float("nan"))) for m in models]
    valid = [(m, v) for m, v in valid if not np.isnan(v)]
    if valid:
        ms = [m for m, _ in valid]
        vs = [v for _, v in valid]
        cs = [COLORS.get(m, "#888") for m in ms]
        bars = ax.barh(ms, vs, color=cs, alpha=0.85, edgecolor="white")
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="RTF = 1")
        vmax = max(vs)
        for bar, v in zip(bars, vs):
            ax.text(v + vmax * 0.025, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=10)
        ax.set_xlabel("RTF  (↓ лучше)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, vmax * 1.20)
        ax.tick_params(axis="y", labelsize=10)
    else:
        ax.text(0.5, 0.5, "RTF не измерен\n(запусти без --skip_rtf)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
    ax.set_title("Real-Time Factor ↓", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    _save("rtf.png")

    # ── WER_n детальный (только нормализованный WER) ──────────
    fig, ax = plt.subplots(figsize=(8, 5))
    wn = [metrics[m]["wer_norm"] for m in models]
    bars = ax.bar(range(len(models)), wn, color=colors, alpha=0.85,
                  edgecolor="white", width=0.5)
    ymax = max(wn)
    for bar, v in zip(bars, wn):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.018,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("WER, %", fontsize=12)
    ax.set_title("WER без пунктуации ↓", fontsize=13, fontweight="bold")
    ax.set_ylim(0, ymax * 1.22)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _save("wer_detail.png")


# ══════════════════════════════════════════════════════════════
#  Точка входа
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",     default="results")
    parser.add_argument("--rtf_samples", type=int, default=5,
                        help="Файлов для замера RTF (default: 5)")
    parser.add_argument("--skip_rtf",    action="store_true",
                        help="Пропустить замер RTF")
    parser.add_argument("--skip_ner",    action="store_true",
                        help="Пропустить NER F1 (ускоряет запуск)")
    args = parser.parse_args()
    cfg  = Config()

    # ── Загрузка предсказаний ─────────────────────────────────
    print("Загрузка файлов предсказаний...")
    raw_data: dict[str, tuple] = {}
    for name, fpath in PRED_FILES.items():
        if not os.path.exists(fpath):
            print(f"  [пропуск] {fpath} не найден")
            continue
        refs, preds = load_pred_file(fpath)
        raw_data[name] = (refs, preds)
        print(f"  {name}: {len(refs)} примеров")

    if not raw_data:
        print("Нет файлов предсказаний. Запусти eval_baseline.py, "
              "eval_finetuned.py, eval_gigaam.py.")
        return

    # ── Постобработка Whisper FT ──────────────────────────────
    if "Whisper FT + PP" in raw_data:
        refs_ft, preds_ft = raw_data["Whisper FT + PP"]
        print("Применение постобработки к Whisper FT...")
        preds_pp = []
        for p in preds_ft:
            try:
                result = postprocess.process(
                    p, restore_punct=False, normalize_nums=False, run_ner=False
                )
                preds_pp.append(result.text_final.lower().strip())
            except Exception:
                preds_pp.append(p)
        raw_data["Whisper FT + PP"] = (refs_ft, preds_pp)
        print("  Постобработка завершена.")

    # ── Общий список refs (для NER референсов) ────────────────
    first_refs = next(iter(raw_data.values()))[0]

    # ── NER для референсов (вычисляется один раз) ─────────────
    ref_ents = None
    if not args.skip_ner:
        print(f"\nNER для референсов ({len(first_refs)} текстов)...")
        ref_ents = _ner_entities_batch(first_refs, "refs")
        ent_count = sum(len(e) for e in ref_ents)
        print(f"  Найдено сущностей: {ent_count} в {len(first_refs)} текстах")

    # ── Вычисление метрик ─────────────────────────────────────
    print("\nВычисление WER / CER / Punct F1 / NER F1...")
    metrics: dict[str, dict] = {}
    for name, (refs, preds) in raw_data.items():
        if not args.skip_ner:
            print(f"  NER predictions для {name}...")
        metrics[name] = compute_all(refs, preds, ref_ents)
        wn  = metrics[name]["wer_norm"]
        wp  = metrics[name]["wer_punct"]
        c   = metrics[name]["cer"]
        pf1 = metrics[name]["punct_f1"]
        nf1 = metrics[name]["ner_f1"]
        nf1_s = f"{nf1:.1f}%" if not np.isnan(nf1) else "  -"
        print(f"  {name}: WER_n={wn:.2f}% WER={wp:.2f}% CER={c:.2f}% "
              f"PunctF1={pf1:.1f}% NER_F1={nf1_s}")

    # ── Замер RTF ─────────────────────────────────────────────
    rtf_vals = {m: float("nan") for m in raw_data}
    if not args.skip_rtf and Path(cfg.OUTPUT_DIR).exists():
        rtf_vals.update(measure_rtf(cfg, n_samples=args.rtf_samples))
    elif args.skip_rtf:
        print("\n[RTF пропущен — флаг --skip_rtf]")
    else:
        print(f"\n[RTF пропущен — адаптеры не найдены: {cfg.OUTPUT_DIR}]")

    # ── Таблица ───────────────────────────────────────────────
    W = 95
    print("\n" + "=" * W)
    print("ИТОГОВЫЕ МЕТРИКИ")
    print(f"  WER   — Word Error Rate с пунктуацией       (меньше = лучше)")
    print(f"  WER_n — WER без пунктуации (только лексика) (меньше = лучше)")
    print(f"  CER   — Character Error Rate                (меньше = лучше)")
    print(f"  PF1   — Punctuation F1                      (больше = лучше)")
    print(f"  NF1   — NER F1 (Natasha, exact match)       (больше = лучше)")
    print(f"  RTF   — Real-Time Factor                    (меньше = лучше)")
    print("-" * W)
    print(f"  {'Система':<22} {'WER':>7} {'WER_n':>7} {'CER':>7} "
          f"{'PF1':>7} {'NF1':>7} {'RTF':>8}")
    print("-" * W)

    baseline_wer = metrics.get("Whisper baseline", {}).get("wer_norm")
    for name, m in metrics.items():
        rtf_s = (f"{rtf_vals[name]:.3f}"
                 if not np.isnan(rtf_vals[name]) else "    -")
        nf1_s = (f"{m['ner_f1']:.1f}%"
                 if not np.isnan(m["ner_f1"]) else "    -")
        delta = ""
        if baseline_wer and name != "Whisper baseline":
            d     = m["wer_norm"] - baseline_wer
            delta = f"  ({d:+.1f}%)"
        print(f"  {name:<22} {m['wer_punct']:>6.2f}%  {m['wer_norm']:>6.2f}%  "
              f"{m['cer']:>6.2f}%  {m['punct_f1']:>6.1f}%  "
              f"{nf1_s:>6}  {rtf_s:>7}{delta}")

    print("=" * W)

    if baseline_wer:
        print("\nОтносительное улучшение WER_n vs Whisper baseline:")
        for name, m in metrics.items():
            if name == "Whisper baseline":
                continue
            rel  = (baseline_wer - m["wer_norm"]) / baseline_wer * 100
            sign = "улучшение" if rel > 0 else "ухудшение"
            print(f"  {name}: {rel:+.1f}% ({sign})")

    # ── Сохранение CSV ────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    rows = [{"system": n, **m, "rtf": rtf_vals[n]} for n, m in metrics.items()]
    csv_path = os.path.join(args.out_dir, "comparison.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nCSV сохранён: {csv_path}")

    # ── Графики ───────────────────────────────────────────────
    print("Построение графиков...")
    plot_all(metrics, rtf_vals, out_dir=args.out_dir)

    # ── Сводка файлов ─────────────────────────────────────────
    print("\n" + "=" * W)
    print("ФАЙЛЫ С РЕЗУЛЬТАТАМИ:")
    print(f"  {csv_path:<45} — все метрики (CSV)")
    for fname, desc in [
        ("wer_norm.png",  "WER без пунктуации"),
        ("wer_detail.png","WER без пунктуации (детальный)"),
        ("cer.png",       "Character Error Rate"),
        ("punct_f1.png",  "Punctuation F1"),
        ("rtf.png",       "Real-Time Factor"),
    ]:
        print(f"  {os.path.join(args.out_dir, fname):<45} — {desc}")
    print(f"  {'predictions_baseline.txt':<45} — предсказания Whisper baseline (2000)")
    print(f"  {'predictions_finetuned.txt':<45} — предсказания Whisper FT (2000)")
    print(f"  {'predictions_gigaam.txt':<45} — предсказания GigaAM v3 (2000)")
    print(f"  {'predictions_borealis.txt':<45} — предсказания Vikhr Borealis (2000)")
    print("=" * W)


if __name__ == "__main__":
    main()
