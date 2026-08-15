import re
import html


def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    text = re.sub(r'<[^>]+>', ' ', text)

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    text = re.sub(r"\s*\\\'\s*", "'", text)

    text = re.sub(r"\s+-\s+", "-", text)

    text = re.sub(r'([a-z])([.!?])([A-Za-z])', r'\1\2 \3', text)

    text = re.sub(r'([a-zA-Z])([;:])([A-Za-z])', r'\1\2 \3', text)

    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    text = re.sub(r'([.,!?;:])([^\s\d"\')\]])', r'\1 \2', text)

    text = re.sub(r'\$\s+(\d+)\s*\.\s*(\d+)', r'$\1.\2', text)

    text = re.sub(r'(\d+)\s*:\s*(\d+)', r'\1:\2', text)

    text = re.sub(r'([!?.]){2,}', r'\1', text)

    text = re.sub(r' {2,}', ' ', text)

    return text.strip()