"""Инференс дообученной модели Whisper-large-v3 + LoRA."""
import argparse
from pathlib import Path

import librosa
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import Config


def load_model(cfg):
    """Загружает базовую модель и применяет обученные LoRA-адаптеры."""
    base_model = WhisperForConditionalGeneration.from_pretrained(
        cfg.MODEL_NAME,
        torch_dtype=torch.bfloat16 if cfg.BF16 else torch.float16,
        device_map="auto",
    )
    # цепляем адаптеры
    model = PeftModel.from_pretrained(base_model, cfg.OUTPUT_DIR)
    model.eval()

    processor = WhisperProcessor.from_pretrained(
        cfg.OUTPUT_DIR if Path(cfg.OUTPUT_DIR).exists() else cfg.MODEL_NAME,
        language=cfg.LANGUAGE,
        task=cfg.TASK,
    )
    return model, processor


def transcribe_file(audio_path, model, processor, cfg):
    """Транскрибирует один аудиофайл, поддерживая длинные записи через чанки."""
    # загрузка и ресемплинг
    audio, sr = librosa.load(audio_path, sr=cfg.SAMPLING_RATE, mono=True)
    chunk_size = cfg.INFERENCE_CHUNK_LENGTH_SEC * cfg.SAMPLING_RATE
    transcripts = []

    # разбиение на 30-секундные чанки
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        if len(chunk) < cfg.SAMPLING_RATE * 0.5:  # пропускаем огрызки <0.5с
            continue

        inputs = processor(
            chunk, sampling_rate=cfg.SAMPLING_RATE, return_tensors="pt"
        )
        input_features = inputs.input_features.to(model.device).to(model.dtype)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language=cfg.LANGUAGE,
                task=cfg.TASK,
                max_new_tokens=440,
                num_beams=5,
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcripts.append(text.strip())

    return " ".join(transcripts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio", required=True, help="Путь к аудиофайлу (wav/mp3/m4a/...)"
    )
    parser.add_argument(
        "--output", default=None, help="Путь для сохранения транскрипта (опционально)"
    )
    args = parser.parse_args()

    cfg = Config()
    print("Загрузка модели...")
    model, processor = load_model(cfg)

    print(f"Транскрибирование: {args.audio}")
    transcript = transcribe_file(args.audio, model, processor, cfg)
    print("\n=== Результат ===")
    print(transcript)

    if args.output:
        Path(args.output).write_text(transcript, encoding="utf-8")
        print(f"\nСохранено в {args.output}")


if __name__ == "__main__":
    main()