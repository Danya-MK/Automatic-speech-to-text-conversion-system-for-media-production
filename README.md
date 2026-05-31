# Automatic Speech-to-Text Conversion System for Media Production

Система автоматического распознавания русской речи на базе **Whisper Large v3** с LoRA-дообучением, постобработкой текста и REST API. Разработана в рамках дипломной работы по специальности «Медиапроизводство».

## Возможности

- **Дообучение** Whisper Large v3 на русском Common Voice через LoRA (PEFT)
- **Постобработка**: восстановление пунктуации, нормализация чисел, извлечение именованных сущностей (NER)
- **REST API** (FastAPI): загрузка аудио, асинхронная обработка, выдача результата в форматах JSON / TXT / SRT
- **Сравнительный бенчмарк** 4 моделей: Whisper baseline, Whisper FT + ПО, GigaAM v3, Vikhr Borealis

## Результаты

### Common Voice Russian (in-domain, n=2000)

| Модель | WER↓ | CER↓ | Punct F1↑ | RTF↓ |
|---|---|---|---|---|
| Whisper baseline | 6.02% | 2.77% | 87.5% | 0.040 |
| GigaAM v3 | 5.31% | 2.11% | 89.4% | 0.067 |
| Vikhr Borealis | 5.09% | 4.78% | 94.5% | 0.102 |
| **Whisper FT + ПО** | **4.94%** | **2.08%** | **91.5%** | **0.056** |

### NER на WikiANN Russian (n=150 синтетических предложений)

| Модель | F1↑ | Precision↑ | Recall↑ |
|---|---|---|---|
| Whisper baseline | 36.5% | 39.1% | 34.2% |
| GigaAM v3 | 36.1% | 39.5% | 33.2% |
| Vikhr Borealis | 34.6% | 40.5% | 30.2% |
| **Whisper FT + ПО** | **37.9%** | **41.8%** | **34.6%** |

> RTF = 0.056 → 2-часовое видео обрабатывается за **~8 минут** (ускорение ×75 по сравнению с ручной расшифровкой).

## Структура репозитория

```
├── api.py                  # REST API (FastAPI)
├── train.py                # Дообучение Whisper + LoRA
├── config.py               # Конфигурация (пути, гиперпараметры)
├── data_utils.py           # Загрузка и предобработка датасета
├── postprocess.py          # Пунктуация, числа, NER
├── inference.py            # Инференс одного файла
├── compare_models.py       # Сравнительный бенчмарк (Common Voice)
├── eval_ner_corpus.py      # NER-бенчмарк (WikiANN + Silero TTS)
├── eval_tedx.py            # Out-of-domain бенчмарк (Multilingual TEDx)
├── eval_baseline.py        # Оценка Whisper baseline
├── eval_finetuned.py       # Оценка дообученной модели
├── eval_gigaam.py          # Оценка GigaAM v3
├── eval_borealis.py        # Оценка Vikhr Borealis
├── requirements.txt
└── results/                # Графики и CSV с результатами
    ├── wer_norm.png
    ├── cer.png
    ├── punct_f1.png
    ├── rtf.png
    ├── wer_detail.png
    ├── ner_f1.png
    ├── ner_precision.png
    ├── ner_recall.png
    ├── tedx_wer.png
    ├── comparison.csv
    ├── ner_comparison.csv
    └── tedx_comparison.csv
```

## Требования

- Python 3.12
- PyTorch 2.7.0 + CUDA 12.8
- NVIDIA GPU (рекомендуется ≥12 ГБ VRAM для обучения, ≥8 ГБ для инференса)
- FFmpeg

## Установка

```bash
git clone https://github.com/Danya-MK/speech-to-text-media-production.git
cd speech-to-text-media-production
pip install -r requirements.txt
```

Для GigaAM v3 — установить отдельно согласно [официальной документации](https://github.com/salute-developers/GigaAM).

## Дообучение

1. Скачать датасет Common Voice Russian и указать путь в `config.py`:
```python
LOCAL_DATASET_PATH = r"/path/to/dataset"
```

2. Запустить обучение:
```bash
python train.py
```

LoRA-адаптеры сохраняются в `./models/whisper-large-v3-ru-lora/`.

Ключевые гиперпараметры (`config.py`):

| Параметр | Значение |
|---|---|
| Base model | openai/whisper-large-v3 |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| Target modules | q_proj, v_proj |
| Batch size | 2 (grad accum ×8 = 16) |
| Learning rate | 1e-4 |
| Epochs | 3 |

## Запуск API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### Пример использования

```bash
# Запуск транскрибирования
curl -X POST http://localhost:8000/transcribe \
     -F "file=@recording.mp3" \
     -F "postprocessing=true"
# → {"task_id": "...", "status": "pending"}

# Получение результата в формате SRT
curl "http://localhost:8000/result/<task_id>?format=srt"
```

### Эндпоинты

| Метод | URL | Описание |
|---|---|---|
| POST | `/transcribe` | Загрузка аудио и запуск транскрибирования |
| GET | `/status/{task_id}` | Статус задачи |
| GET | `/result/{task_id}` | Результат (json / text / srt) |
| GET | `/health` | Состояние сервиса и наличие LoRA |
| GET | `/tasks` | Список всех задач сессии |

## Воспроизведение бенчмарков

```bash
# Сравнение на Common Voice (требует датасет)
python compare_models.py

# NER-бенчмарк (требует Silero TTS + natasha)
python eval_ner_corpus.py

# Out-of-domain тест на TEDx (требует локальный датасет)
python eval_tedx.py --max_samples 200   # быстрый тест
python eval_tedx.py                     # полный прогон
```
