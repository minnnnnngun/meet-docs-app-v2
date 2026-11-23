"""Owner-only recovery journal and durable atomic text exports."""

import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import mkstemp

MAX_CORRECTION_LENGTH = 520
AUDIO_ID = re.compile(r"[0-9a-f]{32}")


def normalize_correction(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("교정 문장을 입력하세요.")
    normalized = " ".join(text.split())
    if not normalized or len(normalized) > MAX_CORRECTION_LENGTH:
        raise ValueError(f"교정 문장을 {MAX_CORRECTION_LENGTH}자 이내로 입력하세요.")
    return normalized


def validate_audio_id(audio_id: str | None) -> None:
    if audio_id is not None and (
        not isinstance(audio_id, str) or not AUDIO_ID.fullmatch(audio_id)
    ):
        raise ValueError("Invalid meeting audio identifier")


def sync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class Export:
    path: Path
    content: str


@dataclass(frozen=True)
class MeetingRecord:
    started_at: datetime
    topic: str
    captions: tuple[str, ...]
    minutes: str
    summarized_count: int
    stem: str | None = None
    verbatim_captions: tuple[str, ...] | None = None
    review_turns_json: tuple[str, ...] = ()
    review_decisions_json: str | None = None
    corrections: dict[int, str] = field(default_factory=dict)
    audio_id: str | None = None


class RecoveryJournal:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir.parent / ".meet-docs-recovery" / "pending.sqlite3"
        self._descriptor: int | None = None

    def exists(self) -> bool:
        return self.path.is_file()

    def acquire(self) -> None:
        if self._descriptor is not None:
            return
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            self.path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR, 0o600
        )
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(descriptor)
            raise
        self._descriptor = descriptor

    def release(self) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        try:
            uri = self.path.resolve().as_uri() + "?mode=rw"
            with closing(sqlite3.connect(uri, uri=True, timeout=2)) as connection:
                connection.execute("PRAGMA synchronous = EXTRA")
                with connection:
                    yield connection
        except sqlite3.Error as exc:
            raise OSError("Could not access the meeting recovery journal") from exc

    def create(self, started_at: datetime, topic: str) -> None:
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """CREATE TABLE meeting (
                    id INTEGER PRIMARY KEY CHECK (id = 1), started_at TEXT NOT NULL,
                    topic TEXT NOT NULL, minutes TEXT NOT NULL DEFAULT '',
                    summarized_count INTEGER NOT NULL DEFAULT 0, stem TEXT, review_decisions_json TEXT,
                    corrections_json TEXT, audio_id TEXT
                    )"""
                )
                connection.execute(
                    "CREATE TABLE captions (sequence INTEGER PRIMARY KEY, line TEXT NOT NULL, "
                    "verbatim_line TEXT, review_turn_json TEXT)"
                )
                connection.execute(
                    "INSERT INTO meeting (id, started_at, topic) VALUES (1, ?, ?)",
                    (started_at.isoformat(), topic),
                )
            sync_directory(self.path.parent)
        except OSError:
            self.clear()
            raise

    def append_caption(
        self,
        sequence: int,
        line: str,
        verbatim_line: str | None = None,
        review_turn_json: str | None = None,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO captions (sequence, line, verbatim_line, review_turn_json) VALUES (?, ?, ?, ?)",
                (sequence, line, verbatim_line, review_turn_json),
            )

    def store_minutes(
        self,
        minutes: str,
        summarized_count: int,
        review_decisions_json: str | None = None,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE meeting SET minutes = ?, summarized_count = ?, review_decisions_json = ? WHERE id = 1",
                (minutes, summarized_count, review_decisions_json),
            )

    def store_stem(self, stem: str) -> None:
        with self._connection() as connection:
            connection.execute("UPDATE meeting SET stem = ? WHERE id = 1", (stem,))

    def store_audio_id(self, audio_id: str) -> None:
        validate_audio_id(audio_id)
        with self._connection() as connection:
            connection.execute(
                "UPDATE meeting SET audio_id = ? WHERE id = 1", (audio_id,)
            )

    def store_corrections(self, corrections: dict[int, str]) -> None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            caption_ids = {
                line
                for (line,) in connection.execute(
                    "SELECT sequence FROM captions WHERE review_turn_json IS NOT NULL"
                )
            }
            if any(
                type(line) is not int or line not in caption_ids for line in corrections
            ):
                raise ValueError("저장된 발언만 교정할 수 있습니다.")
            normalized = {
                line: normalize_correction(correction)
                for line, correction in corrections.items()
            }
            connection.execute(
                "UPDATE meeting SET corrections_json = ? WHERE id = 1",
                (json.dumps(normalized, ensure_ascii=False),),
            )

    def load(self) -> MeetingRecord | None:
        if self.path.stat().st_size == 0:
            return None
        with self._connection() as connection:
            connection.execute("BEGIN")
            if not connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meeting'"
            ).fetchone():
                return None
            meeting_columns = {
                column[1] for column in connection.execute("PRAGMA table_info(meeting)")
            }
            supplemental_columns = ", ".join(
                column if column in meeting_columns else "NULL"
                for column in ("review_decisions_json", "corrections_json", "audio_id")
            )
            row = connection.execute(
                "SELECT started_at, topic, minutes, summarized_count, stem, "
                f"{supplemental_columns} FROM meeting WHERE id = 1"
            ).fetchone()
            if row is None:
                return None
            # Legacy journals store only edited captions.
            caption_columns = {
                column[1]
                for column in connection.execute("PRAGMA table_info(captions)")
            }
            raw_column = (
                "verbatim_line" if "verbatim_line" in caption_columns else "NULL"
            )
            turn_column = (
                "review_turn_json" if "review_turn_json" in caption_columns else "NULL"
            )
            captions = connection.execute(
                f"SELECT sequence, line, {raw_column}, {turn_column} FROM captions ORDER BY sequence"
            ).fetchall()
        (
            started_at,
            topic,
            minutes,
            summarized_count,
            stem,
            review_decisions,
            corrections_json,
            audio_id,
        ) = row
        try:
            start = datetime.fromisoformat(started_at)
        except (TypeError, ValueError) as exc:
            raise OSError("Invalid meeting recovery timestamp") from exc
        if stem is not None:
            date = start.strftime("%Y%m%d_%H%M")
            prefix, _, sequence = stem.rpartition("_")
            if stem != date and (
                prefix != date or not sequence.isascii() or not sequence.isdigit()
            ):
                raise OSError("Invalid meeting recovery filename")
        verbatim = tuple(raw for _, _, raw, _ in captions if raw is not None)
        corrections = {}
        try:
            validate_audio_id(audio_id)
            if corrections_json is not None:
                caption_ids = {
                    sequence for sequence, _, _, turn in captions if turn is not None
                }
                for key, correction in json.loads(corrections_json).items():
                    line = int(key)
                    if line not in caption_ids or line in corrections:
                        raise ValueError("Invalid corrected line")
                    corrections[line] = normalize_correction(correction)
        except (AttributeError, TypeError, ValueError) as exc:
            raise OSError("Invalid meeting recovery correction data") from exc
        return MeetingRecord(
            start,
            topic,
            tuple(line for _, line, _, _ in captions),
            minutes,
            sum(sequence <= summarized_count for sequence, _, _, _ in captions),
            stem,
            verbatim or None,
            tuple(turn for _, _, _, turn in captions if turn is not None),
            review_decisions,
            corrections,
            audio_id,
        )

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
        sync_directory(self.path.parent)


def save_exports(exports: tuple[Export, ...]) -> None:
    if not exports:
        return
    output_dir = exports[0].path.parent
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for export in exports:
        descriptor, name = mkstemp(dir=output_dir, prefix=".meet-docs-", suffix=".tmp")
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(export.content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(export.path)
        finally:
            temporary.unlink(missing_ok=True)
    sync_directory(output_dir)
    sync_directory(output_dir.parent)
