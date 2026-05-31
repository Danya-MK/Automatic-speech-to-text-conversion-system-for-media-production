"""Модуль постобработки транскрипций: пунктуация, нормализация, NER, капитализация."""
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  Нормализация чисел, дат, аббревиатур
# ══════════════════════════════════════════════════════════════

MONTHS_RU = {
    "01": "января", "02": "февраля", "03": "марта",    "04": "апреля",
    "05": "мая",    "06": "июня",    "07": "июля",     "08": "августа",
    "09": "сентября", "10": "октября", "11": "ноября", "12": "декабря",
}

ABBR_DICT = {
    r"\bт\.д\.":      "так далее",
    r"\bт\.е\.":      "то есть",
    r"\bт\.к\.":      "так как",
    r"\bи т\.п\.":    "и тому подобное",
    r"\bи\.о\.":      "исполняющий обязанности",
    r"\bд\.р\.":      "день рождения",
    r"\bул\.":        "улица",
    r"\bпр-т\b":      "проспект",
    r"\bкг\b":        "килограмм",
    r"\bкм\b":        "километров",
    r"\bмлн\b":       "миллионов",
    r"\bмлрд\b":      "миллиардов",
    r"\bтыс\b\.?":    "тысяч",
    r"\bруб\b\.?":    "рублей",
    r"\bгр\b\.?":     "граммов",
    r"\bмин\b\.?(?=\s)": "минут",
    r"\bсек\b\.?(?=\s)": "секунд",
    r"\bчел\b\.?":    "человек",
}


def normalize_numbers(text: str) -> str:
    """Заменяет цифровые числа словами (требует num2words)."""
    try:
        from num2words import num2words

        def _replace(m):
            try:
                return num2words(int(m.group(0)), lang="ru")
            except Exception:
                return m.group(0)

        return re.sub(r"\b\d{1,9}\b", _replace, text)
    except ImportError:
        logger.warning("num2words не установлен: pip install num2words")
        return text


def normalize_dates(text: str) -> str:
    """ДД.ММ.ГГГГ → «12 мая 2024 года»."""
    def _replace(m):
        d, mon, y = m.group(1), m.group(2), m.group(3)
        return f"{int(d)} {MONTHS_RU.get(mon, mon)} {y} года"

    return re.sub(r"\b(\d{1,2})\.(\d{2})\.(\d{4})\b", _replace, text)


def normalize_abbreviations(text: str) -> str:
    """Раскрывает распространённые русские аббревиатуры."""
    for pattern, replacement in ABBR_DICT.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ══════════════════════════════════════════════════════════════
#  Восстановление пунктуации
# ══════════════════════════════════════════════════════════════

_punct_model = None  # ленивая инициализация


def _get_punct_model():
    global _punct_model
    if _punct_model is None:
        from deepmultilingualpunctuation import PunctuationModel
        _punct_model = PunctuationModel(
            model="oliverguhr/fullstop-punctuation-multilang-large"
        )
    return _punct_model


def restore_punctuation(text: str) -> str:
    """
    Снимает пунктуацию с текста и восстанавливает её моделью.
    Работает лучше на тексте без пунктуации, поэтому сначала чистим.
    """
    try:
        model = _get_punct_model()
        # убираем существующую пунктуацию перед подачей в модель
        clean = re.sub(r"[,\.!?;:—\-–]", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return model.restore_punctuation(clean)
    except ImportError:
        logger.warning(
            "deepmultilingualpunctuation не установлен: "
            "pip install deepmultilingualpunctuation"
        )
        return text
    except Exception as e:
        logger.warning(f"Ошибка модели пунктуации: {e}")
        return text


# ══════════════════════════════════════════════════════════════
#  NER (Natasha)
# ══════════════════════════════════════════════════════════════

@dataclass
class Entity:
    text: str
    type: str       # PER, LOC, ORG, DATE, …
    start: int
    stop: int
    normal: Optional[str] = None  # нормализованная форма


_natasha_components = None  # ленивая инициализация


def _get_natasha():
    global _natasha_components
    if _natasha_components is None:
        from natasha import (
            Segmenter, MorphVocab, NewsEmbedding,
            NewsMorphTagger, NewsNERTagger,
        )
        emb = NewsEmbedding()
        _natasha_components = {
            "segmenter":    Segmenter(),
            "morph_vocab":  MorphVocab(),
            "morph_tagger": NewsMorphTagger(emb),
            "ner_tagger":   NewsNERTagger(emb),
        }
    return _natasha_components


def extract_entities(text: str) -> Tuple[str, List[Entity]]:
    """
    Извлекает именованные сущности (NER) через Natasha.
    Возвращает (текст с капитализированными именами, список сущностей).
    """
    try:
        from natasha import Doc
        c = _get_natasha()
    except ImportError:
        logger.warning("natasha не установлена: pip install natasha")
        return text, []

    doc = Doc(text)
    doc.segment(c["segmenter"])
    doc.tag_morph(c["morph_tagger"])
    doc.tag_ner(c["ner_tagger"])

    entities: List[Entity] = []
    chars = list(text)

    for span in doc.spans:
        try:
            span.normalize(c["morph_vocab"])
            normal = span.normal
        except Exception:
            normal = None

        entities.append(Entity(
            text=span.text, type=span.type,
            start=span.start, stop=span.stop, normal=normal,
        ))
        # капитализируем первую букву сущности в исходном тексте
        if 0 <= span.start < len(chars):
            chars[span.start] = chars[span.start].upper()

    return "".join(chars), entities


# ══════════════════════════════════════════════════════════════
#  Капитализация
# ══════════════════════════════════════════════════════════════

def capitalize_sentences(text: str) -> str:
    """Заглавная буква после . ! ? и в начале текста."""
    result = re.sub(
        r"([.!?]\s+)([а-яёa-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        text,
    )
    if result:
        result = result[0].upper() + result[1:]
    return result


# ══════════════════════════════════════════════════════════════
#  Главный пайплайн
# ══════════════════════════════════════════════════════════════

@dataclass
class PostProcessResult:
    text_raw: str
    text_normalized: str
    text_punctuated: str
    text_final: str
    entities: List[Entity] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text_raw":        self.text_raw,
            "text_normalized": self.text_normalized,
            "text_punctuated": self.text_punctuated,
            "text_final":      self.text_final,
            "entities": [
                {"text": e.text, "type": e.type,
                 "start": e.start, "stop": e.stop, "normal": e.normal}
                for e in self.entities
            ],
        }


def process(
    text: str,
    restore_punct: bool = True,
    normalize_nums: bool = True,
    run_ner: bool = True,
) -> PostProcessResult:
    """Полный пайплайн постобработки транскрипции."""

    # 1. Нормализация дат → аббревиатур → чисел
    step1 = normalize_dates(text)
    step1 = normalize_abbreviations(step1)
    if normalize_nums:
        step1 = normalize_numbers(step1)

    # 2. Восстановление / улучшение пунктуации
    step2 = restore_punctuation(step1) if restore_punct else step1

    # 3. NER + капитализация имён
    if run_ner:
        step3, entities = extract_entities(step2)
    else:
        step3, entities = step2, []

    # 4. Капитализация начал предложений
    final = capitalize_sentences(step3)

    return PostProcessResult(
        text_raw=text,
        text_normalized=step1,
        text_punctuated=step2,
        text_final=final,
        entities=entities,
    )
