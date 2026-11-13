"""PII redaction before anything is sent to a model or written to a report."""
import re

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b"), "[IBAN]"),
    (re.compile(r"(?<!\w)(?:\+|00)?\d[\d\s/().-]{7,}\d"), "[PHONE]"),
]


def redact(text: str) -> str:
    for pattern, token in _PATTERNS:
        text = pattern.sub(token, text)
    return text
