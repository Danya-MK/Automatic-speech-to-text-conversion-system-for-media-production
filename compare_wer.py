"""Сравнение WER всех моделей с нормализацией и без.

Читает уже сохранённые файлы predictions_*.txt и считает WER
в двух режимах: как есть (с пунктуацией) и после нормализации.

Использование:
    python compare_wer.py
"""
import re
import os
import evaluate


def normalize(text: str) -> str:
    """Убирает пунктуацию, лишние пробелы; оставляет только слова."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_predictions(path: str):
    """Парсит файл predictions_*.txt → (references, predictions)."""
    refs, preds = [], []
    with open(path, encoding="utf-8") as f:
        content = f.read()

    for block in content.strip().split("\n\n"):
        lines = block.strip().splitlines()
        ref_line  = next((l for l in lines if l.startswith("REF:")),  None)
        pred_line = next((l for l in lines if l.startswith("PRED:")), None)
        if ref_line and pred_line:
            refs.append(ref_line[len("REF:"):].strip())
            preds.append(pred_line[len("PRED:"):].strip())

    return refs, preds


def compute_wer(refs, preds, norm=False):
    metric = evaluate.load("wer")
    if norm:
        refs  = [normalize(r) for r in refs]
        preds = [normalize(p) for p in preds]
    return 100 * metric.compute(predictions=preds, references=refs)


PREDICTION_FILES = {
    "Whisper-large-v3 baseline": "predictions_baseline.txt",
    "Whisper-large-v3 + LoRA":   "predictions_finetuned.txt",
    "GigaAM v3 e2e-rnnt":        "predictions_gigaam.txt",
}


def main():
    results = {}

    for name, filename in PREDICTION_FILES.items():
        if not os.path.exists(filename):
            print(f"[пропуск] {filename} не найден — запусти соответствующий eval_*.py")
            continue

        refs, preds = load_predictions(filename)
        wer_raw  = compute_wer(refs, preds, norm=False)
        wer_norm = compute_wer(refs, preds, norm=True)
        results[name] = (wer_raw, wer_norm, len(refs))
        print(f"Загружено: {filename}  ({len(refs)} примеров)")

    if not results:
        print("Нет файлов с предсказаниями. Сначала запусти eval_*.py --save_predictions")
        return

    # --- Таблица ---
    col = 32
    print(f"\n{'Модель':<{col}} {'WER (с пункт.)':>16} {'WER (без пункт.)':>18} {'Примеров':>10}")
    print("-" * (col + 48))
    for name, (raw, norm, n) in results.items():
        print(f"{name:<{col}} {raw:>14.2f}%  {norm:>16.2f}%  {n:>10}")

    # --- Улучшение относительно baseline ---
    if "Whisper-large-v3 baseline" in results:
        base_raw, base_norm, _ = results["Whisper-large-v3 baseline"]
        print(f"\nОтносительное улучшение WER vs baseline:")
        for name, (raw, norm, _) in results.items():
            if name == "Whisper-large-v3 baseline":
                continue
            rel_raw  = (base_raw  - raw)  / base_raw  * 100
            rel_norm = (base_norm - norm) / base_norm * 100
            print(f"  {name}: {rel_raw:+.1f}% (с пункт.) / {rel_norm:+.1f}% (без пункт.)")

    # --- Примеры нормализации ---
    first_file = next(iter(results))
    filename   = PREDICTION_FILES[first_file]
    refs, preds = load_predictions(filename)
    print(f"\nПример нормализации ({first_file}):")
    for ref, pred in zip(refs[:3], preds[:3]):
        print(f"  REF  orig:  {ref}")
        print(f"  REF  norm:  {normalize(ref)}")
        print(f"  PRED orig:  {pred}")
        print(f"  PRED norm:  {normalize(pred)}")
        print()


if __name__ == "__main__":
    main()
