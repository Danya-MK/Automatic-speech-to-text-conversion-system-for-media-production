"""Оценка WER дообученной модели (Whisper-large-v3 + LoRA) на тестовой выборке."""
import argparse
import os

import evaluate
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from peft import PeftModel
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import Config


def load_test_df(cfg, max_samples: int | None = None) -> pd.DataFrame:
    base = cfg.LOCAL_DATASET_PATH
    for root, dirs, files in os.walk(base):
        if "test.tsv" in files and "clips" in dirs:
            clips_dir = os.path.join(root, "clips")
            df = pd.read_csv(os.path.join(root, "test.tsv"), sep="\t", low_memory=False)
            df = df[["path", "sentence"]].dropna()
            df["path"] = df["path"].apply(lambda p: os.path.join(clips_dir, p))
            if max_samples:
                df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)
            return df
    raise FileNotFoundError(f"test.tsv не найден в {base}")


def run_evaluation(df: pd.DataFrame, model, processor, cfg, batch_size: int = 8):
    wer_metric = evaluate.load("wer")
    predictions, references = [], []
    target_sr = cfg.SAMPLING_RATE

    for i in tqdm(range(0, len(df), batch_size), desc="Inference"):
        batch = df.iloc[i : i + batch_size]
        audio_arrays = []

        for _, row in batch.iterrows():
            try:
                waveform, sr = torchaudio.load(row["path"])
                if sr != target_sr:
                    waveform = T.Resample(sr, target_sr)(waveform)
                audio_arrays.append(waveform.mean(0).numpy())
            except Exception:
                audio_arrays.append(None)

        # фильтруем битые файлы, сохраняем референсы
        valid_audio, valid_refs = [], []
        for arr, (_, row) in zip(audio_arrays, batch.iterrows()):
            if arr is not None:
                valid_audio.append(arr)
                valid_refs.append(row["sentence"].lower().strip())

        if not valid_audio:
            continue

        inputs = processor(
            valid_audio, sampling_rate=target_sr, return_tensors="pt", padding=True
        )
        model_dtype = next(model.parameters()).dtype
        input_features = inputs.input_features.to(device=model.device, dtype=model_dtype)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                language=cfg.LANGUAGE,
                task=cfg.TASK,
                max_new_tokens=225,
            )

        preds = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        predictions.extend([p.lower().strip() for p in preds])
        references.extend(valid_refs)

    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    return wer, predictions, references


def main():
    parser = argparse.ArgumentParser(description="Оценка дообученной модели")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Кол-во тестовых примеров (default: 500)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_predictions", action="store_true",
                        help="Сохранить predictions в predictions_finetuned.txt")
    args = parser.parse_args()

    cfg = Config()

    if not os.path.exists(cfg.OUTPUT_DIR):
        raise FileNotFoundError(
            f"Директория модели не найдена: {cfg.OUTPUT_DIR}\n"
            "Сначала запусти обучение: python train.py"
        )

    print("Загрузка дообученной модели (base + LoRA)...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        cfg.MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, cfg.OUTPUT_DIR)
    model.eval()

    processor = WhisperProcessor.from_pretrained(
        cfg.MODEL_NAME, language=cfg.LANGUAGE, task=cfg.TASK
    )

    print(f"Загрузка тестовой выборки (max {args.max_samples} примеров)...")
    df = load_test_df(cfg, max_samples=args.max_samples)
    print(f"Тестовых примеров: {len(df)}")

    wer, predictions, references = run_evaluation(df, model, processor, cfg, args.batch_size)

    print(f"\n{'='*40}")
    print(f"  WER (дообученная модель): {wer:.2f}%")
    print(f"  Примеров оценено:         {len(predictions)}")
    print(f"{'='*40}")

    # несколько примеров для наглядности
    print("\nПримеры (референс → предсказание):")
    for ref, pred in zip(references[:5], predictions[:5]):
        print(f"  REF:  {ref}")
        print(f"  PRED: {pred}")
        print()

    if args.save_predictions:
        with open("predictions_finetuned.txt", "w", encoding="utf-8") as f:
            for ref, pred in zip(references, predictions):
                f.write(f"REF:  {ref}\nPRED: {pred}\n\n")
        print("Предсказания сохранены в predictions_finetuned.txt")


if __name__ == "__main__":
    main()
