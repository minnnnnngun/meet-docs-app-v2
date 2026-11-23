"""Store confirmed term corrections and suggest similar glossary spellings."""

import os
import re
import sqlite3
import stat
import unicodedata
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from difflib import SequenceMatcher
from hashlib import sha256
from itertools import islice
from pathlib import Path

_TOKEN = re.compile(r"[^\W_]+(?:[+#]{1,2})?")
_ACRONYM = re.compile(r"[A-Z]{2,8}")
_LETTER_NAMES = dict(
    zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        [
            "에이",
            "비",
            "씨",
            "디",
            "이",
            "에프",
            "지",
            "에이치",
            "아이",
            "제이",
            "케이",
            "엘",
            "엠",
            "엔",
            "오",
            "피",
            "큐",
            "알",
            "에스",
            "티",
            "유",
            "브이",
            "더블유",
            "엑스",
            "와이",
            "제트",
        ],
        strict=True,
    )
)
_HANGUL = re.compile(r"[가-힣]")
_TERM_LENGTH = 32
_PHRASE_TOKENS = 3
_HINT_LIMIT = 24
_SUGGESTION_LIMIT = 5


def _phrase(text: str, tokens: list[re.Match[str]]) -> str:
    if not tokens or len(tokens) > _PHRASE_TOKENS:
        return ""
    value = text[tokens[0].start() : tokens[-1].end()]
    if not all(part.isspace() for part in re.split(_TOKEN, value)[1:-1]):
        return ""
    return value if len(value) <= _TERM_LENGTH else ""


def _pronunciation(text: str) -> str:
    return unicodedata.normalize("NFD", "".join(text.casefold().split()))


def _distance(left: str, right: str, limit: int) -> int:
    row = list(range(len(right) + 1))
    for index, source in enumerate(left, start=1):
        previous = row
        row = [index]
        for column, target in enumerate(right, start=1):
            row.append(
                min(
                    row[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (source != target),
                )
            )
        if min(row) > limit:
            return limit + 1
    return row[-1]


def _phonetic_score(left: str, right: str) -> float:
    length = max(len(left), len(right))
    if not length or 5 * abs(len(left) - len(right)) > length:
        return 0.0
    edits = _distance(left, right, length // 5)
    return 1 - edits / length if 5 * edits <= length else 0.0


def _correction_pair(original: str, corrected: str) -> tuple[str, str] | None:
    original = unicodedata.normalize("NFC", original)
    corrected = unicodedata.normalize("NFC", corrected)
    heard = list(_TOKEN.finditer(original))
    intended = list(_TOKEN.finditer(corrected))
    changes = [
        (kind, h1, h2, i1, i2)
        for kind, h1, h2, i1, i2 in SequenceMatcher(
            a=[token.group() for token in heard],
            b=[token.group() for token in intended],
            autojunk=False,
        ).get_opcodes()
        if kind != "equal"
    ]
    if len(changes) != 1:
        return None
    kind, h1, h2, i1, i2 = changes[0]
    if kind != "replace":
        return None
    before = _phrase(original, heard[h1:h2])
    after = _phrase(corrected, intended[i1:i2])
    if not before or not after:
        return None
    if h2 - h1 == i2 - i1 == 1:
        return before, after
    spelled_acronym = any(
        _ACRONYM.fullmatch(short)
        and "".join(_LETTER_NAMES[letter] for letter in short)
        == "".join(spoken.split())
        for short, spoken in ((after, before), (before, after))
    )
    if spelled_acronym or _phonetic_score(
        _pronunciation(before), _pronunciation(after)
    ):
        return " ".join(before.split()), " ".join(after.split())
    return None


def _memory_path(data_dir: Path) -> Path:
    return data_dir / ".meet-docs-corrections" / "pairs.sqlite3"


def _check_location(path: Path, *, create: bool) -> bool:
    if create:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        folder = path.parent.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(folder.st_mode):
        raise OSError("Invalid correction directory")
    if os.name == "posix" and (folder.st_uid != os.getuid() or folder.st_mode & 0o077):
        raise OSError("Correction directory must be owner-only")
    if create:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
    try:
        database = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(database.st_mode) or database.st_nlink != 1:
        raise OSError("Invalid correction database")
    if os.name == "posix" and (
        database.st_uid != os.getuid() or database.st_mode & 0o077
    ):
        raise OSError("Correction database must be owner-only")
    return True


@contextmanager
def _connection(path: Path, *, writable: bool) -> Iterator[sqlite3.Connection]:
    mode = "rw" if writable else "ro"
    try:
        with (
            closing(
                sqlite3.connect(
                    path.absolute().as_uri() + f"?mode={mode}", uri=True, timeout=2
                )
            ) as connection,
            connection,
        ):
            yield connection
    except sqlite3.Error as exc:
        raise OSError("Could not access the correction memory") from exc


def remember_correction(
    data_dir: Path, original: str, corrected: str, *, source: str
) -> None:
    """Replace any previous learning for this confirmed line with one term pair."""
    if not source.strip():
        raise ValueError("A source is required for a correction")
    source_key = sha256(source.encode("utf-8")).hexdigest()
    pair = _correction_pair(original, corrected)
    path = _memory_path(data_dir)
    if not _check_location(path, create=pair is not None):
        return
    with _connection(path, writable=True) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS pairs (
                source_key TEXT PRIMARY KEY, heard_key TEXT NOT NULL,
                heard TEXT NOT NULL, intended TEXT NOT NULL
            )"""
        )
        if pair is None:
            connection.execute("DELETE FROM pairs WHERE source_key = ?", (source_key,))
        else:
            heard, intended = pair
            connection.execute(
                """INSERT INTO pairs VALUES (?, ?, ?, ?)
                ON CONFLICT (source_key) DO UPDATE SET
                    heard_key = excluded.heard_key, heard = excluded.heard,
                    intended = excluded.intended""",
                (source_key, heard.casefold(), heard, intended),
            )


def correction_hints(data_dir: Path) -> tuple[tuple[str, str], ...]:
    """Return a bounded set of unambiguous confirmed hint pairs."""
    path = _memory_path(data_dir)
    if not _check_location(path, create=False):
        return ()
    with _connection(path, writable=False) as connection:
        if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pairs'"
        ).fetchone():
            return ()
        return tuple(
            connection.execute(
                """SELECT MIN(heard), MIN(intended) FROM pairs
                GROUP BY heard_key HAVING COUNT(DISTINCT intended) = 1
                ORDER BY COUNT(*) DESC, heard_key LIMIT ?""",
                (_HINT_LIMIT,),
            )
        )


def suggestions(text: str, words: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Rank glossary candidates with at least 80% similarity between Hangul jamo."""
    glossary: dict[int, dict[str, str]] = {}
    known = set()
    for word in islice(words, 100):
        word = unicodedata.normalize("NFC", word.strip())
        tokens = list(_TOKEN.finditer(word))
        if _phrase(word, tokens) != word or len(_HANGUL.findall(word)) < 3:
            continue
        units = _pronunciation(word)
        known.add(units)
        glossary.setdefault(len(tokens), {}).setdefault(units, word)
    if not glossary:
        return ()
    text = text[:640]
    tokens = list(_TOKEN.finditer(text))
    candidates: dict[tuple[str, str], float] = {}
    for size, targets in glossary.items():
        for start in range(len(tokens) - size + 1):
            heard = _phrase(text, tokens[start : start + size])
            if len(_HANGUL.findall(unicodedata.normalize("NFC", heard))) < 3:
                continue
            units = _pronunciation(heard)
            if units in known:
                continue
            best = 0.0
            choice = None
            for target, spelling in targets.items():
                score = _phonetic_score(units, target)
                if score > best:
                    best, choice = score, spelling
                elif score == best:
                    choice = None
            if choice is not None:
                candidates[heard, choice] = best
    return tuple(
        sorted(candidates, key=lambda pair: (-candidates[pair], pair))[
            :_SUGGESTION_LIMIT
        ]
    )
