"""
Оценка Vikhr Borealis на Common Voice Russian (test, 2000 примеров).

Модель: Vikhrmodels/Borealis
Основана на архитектуре Whisper, дообучена на ~7000 ч русской речи.
Поддерживает пунктуацию в транскрипции.

Использование:
    python eval_borealis.py
    python eval_borealis.py --max_samples 200
"""
import argparse
import json
import os
import re
import time

import evaluate
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer

from config import Config
from data_utils import find_cv_language_root

MODEL_ID  = "Vikhrmodels/Borealis"
PRED_FILE = "predictions_borealis.txt"
RTF_FILE  = "rtf_borealis.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=2000)
    args = parser.parse_args()

    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Загрузка модели ───────────────────────────────────────
    print(f"Загрузка {MODEL_ID}...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device).eval()
    print("  Модель готова.")

    # ── Тестовые данные ───────────────────────────────────────
    cv_root = find_cv_language_root(cfg.LOCAL_DATASET_PATH)
    df      = pd.read_csv(os.path.join(cv_root, "test.tsv"), sep="\t", low_memory=False)
    df      = df[["path", "sentence"]].dropna()
    clips   = os.path.join(cv_root, "clips")
    df["path"] = df["path"].apply(lambda p: os.path.join(clips, p))
    df      = df.head(args.max_samples).reset_index(drop=True)
    print(f"Примеров: {len(df)}")

    # ── Инференс ──────────────────────────────────────────────
    refs, preds = [], []
    errors = 0
    total_audio_sec = 0.0
    total_infer_sec = 0.0

    for i, row in df.iterrows():
        try:
            waveform, sr = torchaudio.load(row["path"])
            if sr != 16000:
                waveform = T.Resample(sr, 16000)(waveform)
            arr = waveform.mean(0).numpy()
            audio_dur = len(arr) / 16000
            total_audio_sec += audio_dur

            proc = extractor(
                arr,
                sampling_rate=16000,
                padding="max_length",
                max_length=480_000,
                return_attention_mask=True,
                return_tensors="pt",
            )
            mel      = proc.input_features.squeeze(0).to(device)
            att_mask = proc.attention_mask.squeeze(0).to(device)

            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    mel=mel,
                    att_mask=att_mask,
                    max_new_tokens=350,
                    do_sample=False,
                )
            total_infer_sec += time.perf_counter() - t0

            # generate может вернуть строку (кастомный код) или тензор токенов
            if isinstance(out, str):
                transcription = out.lower().strip()
            elif isinstance(out, list) and out and isinstance(out[0], str):
                transcription = out[0].lower().strip()
            else:
                transcription = tokenizer.decode(
                    out[0], skip_special_tokens=True
                ).lower().strip()

            refs.append(row["sentence"])
            preds.append(transcription)

        except Exception as e:
            errors += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(df)}")

    print(f"  Ошибок пропущено: {errors}")

    # ── RTF ───────────────────────────────────────────────────
    rtf = total_infer_sec / total_audio_sec if total_audio_sec > 0 else float("nan")
    print(f"  RTF = {rtf:.4f}  (инференс {total_infer_sec:.1f}с / аудио {total_audio_sec:.1f}с)")
    with open(RTF_FILE, "w") as f:
        json.dump({"model": "Vikhr Borealis", "rtf": rtf,
                   "total_audio_sec": total_audio_sec,
                   "total_infer_sec": total_infer_sec}, f)

    # ── WER / CER ─────────────────────────────────────────────
    def normalize(text):
        t = text.lower().strip()
        t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
        return re.sub(r"\s+", " ", t).strip()

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    wer_p = 100 * wer_metric.compute(predictions=preds, references=refs)
    wer_n = 100 * wer_metric.compute(
        predictions=[normalize(p) for p in preds],
        references=[normalize(r) for r in refs],
    )
    cer_v = 100 * cer_metric.compute(predictions=preds, references=refs)

    print(f"\nРезультаты [Vikhr Borealis]:")
    print(f"  WER (с пункт.)   = {wer_p:.2f}%")
    print(f"  WER (без пункт.) = {wer_n:.2f}%")
    print(f"  CER              = {cer_v:.2f}%")

    # ── Сохранение предсказаний ───────────────────────────────
    with open(PRED_FILE, "w", encoding="utf-8") as f:
        for r, p in zip(refs, preds):
            f.write(f"REF:  {r}\nPRED: {p}\n\n")
    print(f"\nСохранено: {PRED_FILE}  ({len(refs)} примеров)")


if __name__ == "__main__":
    main()
