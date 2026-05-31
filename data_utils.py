"""Загрузка и препроцессинг локальной копии Common Voice."""
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
from datasets import Dataset


def find_cv_language_root(base_path: str) -> str:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Путь не существует: {base_path}")
    for root, dirs, files in os.walk(base_path):
        if "train.tsv" in files and "clips" in dirs:
            return root
    raise FileNotFoundError(
        f"Не найден train.tsv с папкой clips/ внутри {base_path}. "
        "Проверь, что датасет распакован."
    )


def _read_split(cv_root: str, split: str, max_samples: int | None = None) -> pd.DataFrame:
    clips_dir = os.path.join(cv_root, "clips")
    tsv_path = os.path.join(cv_root, f"{split}.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Нет файла {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    df = df[["path", "sentence"]].dropna()
    df["path"] = df["path"].apply(lambda p: os.path.join(clips_dir, p))
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)
    return df


def _make_generator(df, feature_extractor, tokenizer, target_sr, max_length_sec):
    """Возвращает генератор примеров; работает в главном процессе (без IPC)."""
    def _gen():
        import torchaudio
        import torchaudio.transforms as T

        for _, row in df.iterrows():
            try:
                waveform, sr = torchaudio.load(row["path"])
                if sr != target_sr:
                    waveform = T.Resample(sr, target_sr)(waveform)
                arr = waveform.mean(0).numpy()
                dur = len(arr) / target_sr
                if dur >= max_length_sec:
                    continue
                feat = feature_extractor(arr, sampling_rate=target_sr).input_features[0]
                ids = tokenizer(row["sentence"]).input_ids
                yield {"input_features": feat, "labels": ids, "input_length": dur}
            except Exception:
                continue

    return _gen


def load_and_prepare_dataset(cfg, feature_extractor, tokenizer):
    cv_root = find_cv_language_root(cfg.LOCAL_DATASET_PATH)
    print(f"Найден корень датасета: {cv_root}")

    train_df = _read_split(cv_root, cfg.TRAIN_SPLIT, max_samples=None)   # весь датасет
    eval_df = _read_split(cv_root, cfg.EVAL_SPLIT, max_samples=1000)    # 1k для быстрой оценки при обучении
    print(f"Отбор: train={len(train_df)}, eval={len(eval_df)}")

    print("Обработка train (в главном процессе)...")
    train_ds = Dataset.from_generator(
        _make_generator(train_df, feature_extractor, tokenizer, cfg.SAMPLING_RATE, cfg.MAX_AUDIO_LENGTH_SEC)
    )
    print(f"Train готов: {len(train_ds)} примеров")

    print("Обработка eval (в главном процессе)...")
    eval_ds = Dataset.from_generator(
        _make_generator(eval_df, feature_extractor, tokenizer, cfg.SAMPLING_RATE, cfg.MAX_AUDIO_LENGTH_SEC)
    )
    print(f"Eval готов: {len(eval_ds)} примеров")

    return train_ds, eval_ds


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Паддинг для аудиофич и токенов одновременно."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
