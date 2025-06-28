import re
from jiwer import wer


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def word_error_rate(reference: str, hypothesis: str, preprocess: bool = True) -> float:
    if preprocess:
        reference = preprocess_text(reference)
        hypothesis = preprocess_text(hypothesis)
    return wer(reference, hypothesis)
