"""Дообучение Whisper-large-v3 на русской части Common Voice через LoRA."""
import os

import evaluate
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from config import Config
from data_utils import DataCollatorSpeechSeq2SeqWithPadding, load_and_prepare_dataset


def main():
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 1. Процессор (feature extractor + tokenizer)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(
        cfg.MODEL_NAME, language=cfg.LANGUAGE, task=cfg.TASK
    )
    processor = WhisperProcessor.from_pretrained(
        cfg.MODEL_NAME, language=cfg.LANGUAGE, task=cfg.TASK
    )

    # 2. Модель в 8-битах для экономии памяти
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.MODEL_NAME,
        load_in_8bit=True,           # квантование для базовых весов
        device_map="auto",
    )
    # фиксируем язык/задачу
    model.generation_config.language = cfg.LANGUAGE
    model.generation_config.task = cfg.TASK
    model.generation_config.forced_decoder_ids = None
    # подготовка к k-bit обучению (отключает кэш, нужно для checkpointing)
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA-адаптеры
    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        target_modules=cfg.LORA_TARGET_MODULES,
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # покажет, что обучаемо ~1% параметров

    # 4. Данные
    train_ds, eval_ds = load_and_prepare_dataset(cfg, feature_extractor, tokenizer)
    print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 5. Метрика WER
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # -100 нужно заменить на pad_token_id для декодинга
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # 6. Аргументы тренинга
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS,
        learning_rate=cfg.LEARNING_RATE,
        warmup_steps=cfg.WARMUP_STEPS,
        num_train_epochs=cfg.NUM_EPOCHS,
        gradient_checkpointing=cfg.GRADIENT_CHECKPOINTING,
        bf16=cfg.BF16,
        fp16=cfg.FP16,
        eval_strategy="steps",
        eval_steps=cfg.EVAL_STEPS,
        save_steps=cfg.SAVE_STEPS,
        logging_steps=cfg.LOGGING_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        # критично: 8-bit Adam через accelerate
        optim="adamw_bnb_8bit" if cfg.USE_8BIT_OPTIMIZER else "adamw_torch",
        # для PEFT нужно отключить remove_unused_columns
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # 7. Обучение
    trainer.train()

    # 8. Сохранение только LoRA-адаптеров (несколько мегабайт)
    model.save_pretrained(cfg.OUTPUT_DIR)
    processor.save_pretrained(cfg.OUTPUT_DIR)
    print(f"LoRA-адаптеры сохранены в {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()