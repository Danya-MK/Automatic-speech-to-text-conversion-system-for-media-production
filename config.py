"""Конфигурация для дообучения Whisper-large-v3 на Common Voice (русский)."""

class Config:
    # Модель
    MODEL_NAME = "openai/whisper-large-v3"
    LANGUAGE = "russian"
    TASK = "transcribe"

    # Данные
    LOCAL_DATASET_PATH = r"C:\whisper-finetune\dataset"  # <-- добавить
    # DATASET_NAME и DATASET_CONFIG больше не нужны, можно удалить
    TRAIN_SPLIT = "train"
    EVAL_SPLIT = "test"
    MAX_AUDIO_LENGTH_SEC = 30
    SAMPLING_RATE = 16000

    # LoRA
    LORA_R = 32                 # ранг адаптера
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # классика для трансформеров

    # Обучение
    OUTPUT_DIR = "./models/whisper-large-v3-ru-lora"
    BATCH_SIZE = 2              # под 12 ГБ VRAM
    GRAD_ACCUM_STEPS = 8        # эффективный batch = 16
    LEARNING_RATE = 1e-4        # для LoRA можно выше, чем для full FT
    NUM_EPOCHS = 3
    WARMUP_STEPS = 100
    EVAL_STEPS = 500
    SAVE_STEPS = 500
    LOGGING_STEPS = 25

    # Оптимизация памяти
    USE_8BIT_OPTIMIZER = True
    GRADIENT_CHECKPOINTING = True
    BF16 = True                 # для RTX 5070 (Blackwell) bf16 хорошо поддерживается
    FP16 = False                # либо bf16, либо fp16

    # Инференс
    INFERENCE_CHUNK_LENGTH_SEC = 30
    INFERENCE_BATCH_SIZE = 1