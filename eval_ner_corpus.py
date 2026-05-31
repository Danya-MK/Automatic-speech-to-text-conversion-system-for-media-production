"""
Оценка качества NER в ASR-пайплайне на WikiANN Russian.

Подход:
  1. WikiANN Russian (test) — предложения с размеченными сущностями PER/ORG/LOC
  2. Silero TTS синтезирует аудио из этих предложений (ru_v3, 24kHz→16kHz)
  3. Три ASR-системы транскрибируют аудио
  4. Natasha NER извлекает сущности из транскрипций
  5. Сравниваем с золотой разметкой WikiANN → NER F1 (exact match)

Логика: если ASR допускает меньше ошибок → NER агент видит правильный текст →
выше NER F1. Метрика показывает downstream-качество пайплайна.

Использование:
    python eval_ner_corpus.py
    python eval_ner_corpus.py --max_sentences 50    # быстрый тест
    python eval_ner_corpus.py --models baseline ft  # без GigaAM
"""
import argparse
import os
import re
import tempfile
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as TT

matplotlib.rcParams["font.family"] = "DejaVu Sans"

import postprocess
from config import Config

COLORS = {
    "Whisper baseline": "#4C72B0",
    "GigaAM v3":        "#55A868",
    "Vikhr Borealis":   "#C44E52",
    "Whisper FT + PP":  "#DD8452",
}

# ══════════════════════════════════════════════════════════════
#  WikiANN — загрузка и фильтрация
# ══════════════════════════════════════════════════════════════

_WIKI_NOISE = re.compile(r"==|перенаправление|redirect|\[\[|\]\]", re.I)


def _parse_spans(spans: list) -> frozenset:
    """'PER: Иван Петров' → frozenset{('иван петров', 'PER')}"""
    result = set()
    for s in spans:
        if ": " in s:
            etype, etext = s.split(": ", 1)
            result.add((etext.lower().strip(), etype.strip()))
    return frozenset(result)


def _clean_tokens(tokens: list) -> str:
    """Соединяет токены, убирает лишние пробелы перед знаками пунктуации."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,\.!?;:\)\]»])", r"\1", text)
    text = re.sub(r"([\(\[«])\s+", r"\1", text)
    return text.strip()


def load_wikiann(max_sentences: int = None, min_entities: int = 1,
                 min_words: int = 5, max_entity_tokens: int = None):
    """
    Загружает тестовый сплит WikiANN Russian.
    Фильтрует: Wikipedia-мусор, слишком короткие строки,
    и (опционально) предложения с длинными сущностями.

    max_entity_tokens — максимальная длина любой сущности в предложении
    (1 = только однословные, 2 = до двух слов, None = без ограничений).
    """
    from datasets import load_dataset

    print("Загрузка WikiANN Russian (test)...")
    ds = load_dataset("wikiann", "ru", split="test", trust_remote_code=False)

    limit_str = (f", макс. сущность ≤ {max_entity_tokens} слов"
                 if max_entity_tokens else "")
    print(f"  Фильтр: ≥{min_entities} сущности, ≥{min_words} слов{limit_str}")

    sentences, gold_ents = [], []
    for ex in ds:
        spans = ex.get("spans", [])
        if len(spans) < min_entities:
            continue
        text = _clean_tokens(ex["tokens"])
        if _WIKI_NOISE.search(text):
            continue
        words = [w for w in text.split() if re.search(r"\w", w)]
        if len(words) < min_words:
            continue
        # Фильтр по длине сущностей
        entities = _parse_spans(spans)
        if max_entity_tokens:
            if any(len(etext.split()) > max_entity_tokens
                   for etext, _ in entities):
                continue
        sentences.append(text)
        gold_ents.append(entities)
        if max_sentences and len(sentences) >= max_sentences:
            break

    print(f"  Загружено: {len(sentences)} предложений")
    total_ents = sum(len(e) for e in gold_ents)
    print(f"  Золотых сущностей: {total_ents} "
          f"({total_ents/len(sentences):.1f} на предложение)")
    return sentences, gold_ents


# ══════════════════════════════════════════════════════════════
#  Silero TTS
# ══════════════════════════════════════════════════════════════

def load_tts():
    print("Загрузка Silero TTS (ru_v3)...")
    model, _ = torch.hub.load(
        "snakers4/silero-models", "silero_tts",
        language="ru", speaker="ru_v3",
    )
    model.to("cpu")
    print("  TTS готова.")
    return model


def synthesize(tts_model, text: str, target_sr: int = 16000) -> np.ndarray:
    """Синтез аудио: 24kHz → ресэмпл до 16kHz → float32 numpy."""
    try:
        audio = tts_model.apply_tts(
            text=text, speaker="aidar", sample_rate=24000,
            put_accent=True, put_yo=True,
        )
        # audio: 1-D CPU tensor, 24kHz
        audio_2d = audio.unsqueeze(0)                   # (1, T)
        resampler = TT.Resample(24000, target_sr)
        audio_16k = resampler(audio_2d).squeeze(0)      # (T',)
        return audio_16k.numpy().astype(np.float32)
    except Exception as e:
        print(f"  TTS ошибка: {e}")
        return np.zeros(target_sr, dtype=np.float32)    # 1 сек тишины


# ══════════════════════════════════════════════════════════════
#  NER: Natasha
# ══════════════════════════════════════════════════════════════

def natasha_ner(text: str) -> frozenset:
    """Возвращает frozenset{(текст_нижний, тип)} для всех сущностей."""
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


def ner_f1(gold_list: list, pred_list: list) -> dict:
    """Entity-level F1 (exact match на текст + тип)."""
    tp = fp = fn = 0
    for gold, pred_text in zip(gold_list, pred_list):
        pred_ents = natasha_ner(pred_text)
        tp += len(gold & pred_ents)
        fp += len(pred_ents - gold)
        fn += len(gold - pred_ents)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "f1":        round(100 * f1,   2),
        "precision": round(100 * prec, 2),
        "recall":    round(100 * rec,  2),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ══════════════════════════════════════════════════════════════
#  ASR инференс
# ══════════════════════════════════════════════════════════════

def _whisper_infer(audio_arrays: list, cfg, use_lora: bool) -> list:
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
    dtype = next(model.parameters()).dtype
    preds = []
    for i, arr in enumerate(audio_arrays, 1):
        inp  = processor(arr, sampling_rate=16000, return_tensors="pt")
        feat = inp.input_features.to(model.device, dtype=dtype)
        with torch.no_grad():
            ids = model.generate(feat, language=cfg.LANGUAGE,
                                 task=cfg.TASK, max_new_tokens=225)
        preds.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
        if i % 50 == 0:
            print(f"    {i}/{len(audio_arrays)}")
    del model
    torch.cuda.empty_cache()
    return preds


def _borealis_infer(audio_arrays: list) -> list:
    """Инференс Vikhr Borealis (Whisper-энкодер + Qwen2.5-0.5B декодер)."""
    from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer
    print("  Загрузка Vikhr Borealis...")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AutoFeatureExtractor.from_pretrained("Vikhrmodels/Borealis", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Vikhrmodels/Borealis", trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        "Vikhrmodels/Borealis", trust_remote_code=True, torch_dtype=torch.float16
    ).to(device).eval()
    preds = []
    for i, arr in enumerate(audio_arrays, 1):
        try:
            proc = extractor(arr, sampling_rate=16000, padding="max_length",
                             max_length=480_000, return_attention_mask=True,
                             return_tensors="pt")
            mel = proc.input_features.squeeze(0).to(device)
            att = proc.attention_mask.squeeze(0).to(device)
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
        except Exception as e:
            preds.append("")
        if i % 50 == 0:
            print(f"    {i}/{len(audio_arrays)}")
    del model
    torch.cuda.empty_cache()
    return preds


def _gigaam_infer(audio_arrays: list) -> list | None:
    try:
        import gigaam
    except ImportError:
        print("  [пропуск] gigaam не установлен")
        return None
    import soundfile as sf
    print("  Загрузка GigaAM v3...")
    model = gigaam.load_model("v3_e2e_rnnt")
    preds = []
    for i, arr in enumerate(audio_arrays, 1):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, arr, 16000, subtype="PCM_16")
        tmp.close()
        text = model.transcribe(tmp.name)
        os.unlink(tmp.name)
        preds.append(text if isinstance(text, str) else str(text))
        if i % 50 == 0:
            print(f"    {i}/{len(audio_arrays)}")
    return preds


# ══════════════════════════════════════════════════════════════
#  Графики
# ══════════════════════════════════════════════════════════════

def plot_ner(results: dict, n_sentences: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    models = list(results.keys())
    colors = [COLORS.get(m, "#888") for m in models]

    plots = [
        ("f1",        "ner_f1.png",        "NER F1 на WikiANN Russian ↑",        "F1, %"),
        ("precision", "ner_precision.png",  "NER Precision на WikiANN Russian ↑", "Precision, %"),
        ("recall",    "ner_recall.png",     "NER Recall на WikiANN Russian ↑",    "Recall, %"),
    ]

    for metric, fname, title, ylabel in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        vals = [results[m][metric] for m in models]
        bars = ax.bar(range(len(models)), vals, color=colors, alpha=0.85,
                      edgecolor="white", width=0.5)
        ymax = max(vals) if vals else 1
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.02,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=11)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=10, rotation=15, ha="right")
        ax.set_ylabel(ylabel + ", ↑ лучше", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, min(105, ymax * 1.25))
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Сохранён: {path}")


# ══════════════════════════════════════════════════════════════
#  Точка входа
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",        default="results")
    parser.add_argument("--max_sentences",  type=int, default=150,
                        help="Максимум предложений из WikiANN (default: 150)")
    parser.add_argument("--min_entities",      type=int, default=1,
                        help="Минимум сущностей в предложении (default: 1)")
    parser.add_argument("--max_entity_tokens", type=int, default=None,
                        help="Макс. длина сущности в токенах (1=однословные, 2=до двух слов)")
    parser.add_argument("--models",         nargs="+",
                        choices=["baseline", "ft", "gigaam", "borealis"],
                        default=["baseline", "ft", "gigaam", "borealis"])
    args = parser.parse_args()
    cfg  = Config()

    # ── 1. Данные ─────────────────────────────────────────────
    sentences, gold_ents = load_wikiann(
        max_sentences=args.max_sentences,
        min_entities=args.min_entities,
        max_entity_tokens=args.max_entity_tokens,
    )

    # ── 2. TTS ────────────────────────────────────────────────
    print(f"\nСинтез аудио Silero TTS ({len(sentences)} предложений)...")
    tts    = load_tts()
    audios = []
    for i, sent in enumerate(sentences, 1):
        audios.append(synthesize(tts, sent))
        if i % 25 == 0:
            print(f"  {i}/{len(sentences)}")
    del tts
    torch.cuda.empty_cache()
    total_min = sum(len(a) / 16000 for a in audios) / 60
    print(f"  Синтезировано: {len(audios)} аудио, ~{total_min:.1f} мин")

    # ── 3. ASR ────────────────────────────────────────────────
    asr_preds: dict[str, list] = {}

    if "baseline" in args.models:
        print("\n[ASR] Whisper baseline")
        asr_preds["Whisper baseline"] = _whisper_infer(audios, cfg, use_lora=False)

    if "ft" in args.models:
        if not Path(cfg.OUTPUT_DIR).exists():
            print(f"\n[ASR] Whisper FT: [пропуск] адаптеры не найдены: {cfg.OUTPUT_DIR}")
        else:
            print("\n[ASR] Whisper FT + постобработка")
            raw = _whisper_infer(audios, cfg, use_lora=True)
            pp  = []
            for p in raw:
                try:
                    pp.append(postprocess.process(
                        p, restore_punct=False, normalize_nums=False, run_ner=False
                    ).text_final)
                except Exception:
                    pp.append(p)
            asr_preds["Whisper FT + PP"] = pp

    if "gigaam" in args.models:
        print("\n[ASR] GigaAM v3")
        preds = _gigaam_infer(audios)
        if preds:
            asr_preds["GigaAM v3"] = preds

    if "borealis" in args.models:
        print("\n[ASR] Vikhr Borealis")
        asr_preds["Vikhr Borealis"] = _borealis_infer(audios)

    if not asr_preds:
        print("Нет предсказаний.")
        return

    # ── 4. NER F1 ─────────────────────────────────────────────
    print("\nNER F1 (Natasha NER на ASR-выводе)...")
    results: dict[str, dict] = {}
    for name, preds in asr_preds.items():
        print(f"  {name}...", end=" ", flush=True)
        m = ner_f1(gold_ents[:len(preds)], preds)
        results[name] = m
        print(f"F1={m['f1']:.1f}%  P={m['precision']:.1f}%  R={m['recall']:.1f}%"
              f"  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    # ── 5. Таблица ────────────────────────────────────────────
    W = 75
    print("\n" + "=" * W)
    print(f"NER F1 — WikiANN Russian ({len(sentences)} предложений, "
          f"TTS → ASR → Natasha)")
    print("-" * W)
    print(f"  {'Система':<22} {'F1':>7} {'Precision':>10} {'Recall':>8}  "
          f"{'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * W)
    baseline_f1 = results.get("Whisper baseline", {}).get("f1")
    for name, m in results.items():
        delta = ""
        if baseline_f1 and name != "Whisper baseline":
            d     = m["f1"] - baseline_f1
            delta = f"  ({d:+.1f}%)"
        print(f"  {name:<22} {m['f1']:>6.1f}%  {m['precision']:>8.1f}%  "
              f"{m['recall']:>6.1f}%  {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}{delta}")
    print("=" * W)

    # ── 6. Сохранение ─────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    rows = [{"system": n, **m} for n, m in results.items()]
    csv_path = os.path.join(args.out_dir, "ner_comparison.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.2f")
    print(f"\nCSV: {csv_path}")

    print("Построение графиков...")
    plot_ner(results, len(sentences), out_dir=args.out_dir)

    print(f"\nГотово. Результаты в папке: {args.out_dir}/")
    print(f"  {csv_path}")
    print(f"  {os.path.join(args.out_dir, 'ner_comparison.png')}")


if __name__ == "__main__":
    main()
