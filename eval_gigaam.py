"""Оценка WER модели GigaAM v2 на тестовой выборке Common Voice Russian.

Установка:
    pip install gigaam
"""
import argparse
import os
import tempfile

import evaluate
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

try:
    import gigaam
except ImportError:
    raise SystemExit(
        "GigaAM не установлен.\n"
        "Установите командой: pip install gigaam\n"
        "Документация: https://github.com/salute-developers/GigaAM"
    )

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


def audio_to_wav_temp(audio_path: str, target_sr: int = 16000) -> str:
    """Конвертирует аудио в WAV 16kHz моно во временный файл."""
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = T.Resample(sr, target_sr)(waveform)
    waveform = waveform.mean(0, keepdim=True)  # → моно

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    torchaudio.save(tmp.name, waveform, target_sr)
    return tmp.name


def main():
    parser = argparse.ArgumentParser(description="Оценка GigaAM v2 CTC/RNNT")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Кол-во тестовых примеров (default: 500)")
    available = list(gigaam._MODEL_HASHES.keys())
    parser.add_argument("--model_type", choices=available, default="v3_e2e_rnnt",
                        help=f"Тип модели GigaAM. Доступные: {available}")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Сохранить predictions в predictions_gigaam.txt")
    args = parser.parse_args()

    cfg = Config()

    print(f"Загрузка GigaAM {args.model_type}...")
    model = gigaam.load_model(args.model_type)

    print(f"Загрузка тестовой выборки (max {args.max_samples} примеров)...")
    df = load_test_df(cfg, max_samples=args.max_samples)
    print(f"Тестовых примеров: {len(df)}")

    wer_metric = evaluate.load("wer")
    predictions, references = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        tmp_path = None
        try:
            tmp_path = audio_to_wav_temp(row["path"], target_sr=16000)
            text = model.transcribe(tmp_path)
            if isinstance(text, list):
                text = text[0]
            predictions.append(str(text).lower().strip())
            references.append(row["sentence"].lower().strip())
        except Exception as e:
            # пропускаем битые файлы
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    wer = 100 * wer_metric.compute(predictions=predictions, references=references)

    print(f"\n{'='*40}")
    print(f"  Модель:           GigaAM {args.model_type}")
    print(f"  WER:              {wer:.2f}%")
    print(f"  Примеров оценено: {len(predictions)}")
    print(f"{'='*40}")

    print("\nПримеры (референс → предсказание):")
    for ref, pred in zip(references[:5], predictions[:5]):
        print(f"  REF:  {ref}")
        print(f"  PRED: {pred}")
        print()

    if args.save_predictions:
        with open("predictions_gigaam.txt", "w", encoding="utf-8") as f:
            for ref, pred in zip(references, predictions):
                f.write(f"REF:  {ref}\nPRED: {pred}\n\n")
        print("Предсказания сохранены в predictions_gigaam.txt")

    # Итоговая таблица для сравнения
    print("\n=== Сравнение (запусти все три скрипта) ===")
    print(f"  Whisper-large-v3 (baseline):  10.30%  [уже измерено]")
    print(f"  Whisper-large-v3 + LoRA:       7.95%  [уже измерено]")
    print(f"  GigaAM {args.model_type}:     {wer:.2f}%  [только что]")


if __name__ == "__main__":
    main()
