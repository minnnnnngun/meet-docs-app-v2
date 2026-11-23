"""Local glossary storage. Only keys are passed to the meeting session."""

import json
from pathlib import Path

from meeting_storage import Export, save_exports

GLOSSARY_FILE = "team_glossary.json"


def parse_glossary(content: str) -> dict[str, str]:
    entries = json.loads(content)
    if not isinstance(entries, dict) or len(entries) > 100:
        raise ValueError("단어와 설명을 담은 JSON 객체에 최대 100개를 넣으세요.")
    words: dict[str, str] = {}
    for raw_word, raw_note in entries.items():
        word = raw_word.strip()
        if (
            not word
            or len(word) > 80
            or not word.isprintable()
            or not isinstance(raw_note, str)
            or len(raw_note) > 320
            or word in words
        ):
            raise ValueError(
                "서로 다른 단어(80자 이하)와 설명(320자 이하)을 입력하세요."
            )
        words[word] = raw_note
    return words


def read_glossary(data_dir: Path) -> dict[str, str]:
    try:
        return parse_glossary((data_dir / GLOSSARY_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def save_glossary(data_dir: Path, content: str) -> None:
    words = parse_glossary(content)
    save_exports(
        (
            Export(
                data_dir / GLOSSARY_FILE,
                json.dumps(words, ensure_ascii=False, indent=2) + "\n",
            ),
        )
    )
