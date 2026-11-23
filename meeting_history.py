"""Read archived meetings directly from the app's exported files."""

import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

from meeting_storage import Export

MINUTES = "회의록.md"
TRANSCRIPT = "전체녹취.txt"
VERBATIM = "전사원문.txt"
EXPORT_KINDS = (MINUTES, TRANSCRIPT, VERBATIM)
STEM = r"(?P<stamp>[0-9]{8}_[0-9]{4})(?:_(?P<sequence>[1-9][0-9]*))?"
ARCHIVE_FILENAME = re.compile(
    rf"(?P<stem>{STEM})_(?P<kind>{'|'.join(map(re.escape, EXPORT_KINDS))})"
)


@dataclass(frozen=True)
class ArchivedMeeting:
    stem: str
    topic: str
    started_at: datetime
    round_number: int
    has_minutes: bool
    has_transcript: bool
    has_verbatim: bool = False


def _open_export(path: Path) -> TextIO:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise OSError("Meeting export must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("Meeting export must be a regular file")
        return os.fdopen(descriptor, encoding="utf-8-sig")
    except (OSError, ValueError):
        os.close(descriptor)
        raise


def _export_header(path: Path, stamp: datetime) -> tuple[datetime, str]:
    try:
        with _open_export(path) as stream:
            for _ in range(3):
                line = stream.readline(1024).strip()
                if line.startswith("회의 시작 "):
                    started_at = datetime.fromisoformat(line.removeprefix("회의 시작 "))
                    topic = stream.readline(1024).strip()
                    if started_at.replace(second=0, microsecond=0) != stamp:
                        started_at = stamp
                    return started_at, (
                        topic.removeprefix("회의 주제 ").strip()
                        if topic.startswith("회의 주제 ")
                        else ""
                    )
    except (OSError, ValueError):
        pass
    return stamp, ""


def list_meetings(output_dir: Path) -> tuple[ArchivedMeeting, ...]:
    """Return newest meetings first, numbering records with the same topic."""
    if output_dir.is_symlink():
        return ()
    files: dict[str, set[str]] = {}
    try:
        with os.scandir(output_dir) as entries:
            for entry in entries:
                match = ARCHIVE_FILENAME.fullmatch(entry.name)
                if match and entry.is_file(follow_symlinks=False):
                    files.setdefault(match["stem"], set()).add(match["kind"])
    except (FileNotFoundError, NotADirectoryError):
        return ()

    records = []
    for stem, kinds in files.items():
        match = re.fullmatch(STEM, stem)
        assert match is not None
        try:
            stamp = datetime.strptime(match["stamp"], "%Y%m%d_%H%M")
        except ValueError:
            continue
        headers = [
            _export_header(output_dir / f"{stem}_{kind}", stamp)
            for kind in EXPORT_KINDS
            if kind in kinds
        ]
        topic = next((topic for _, topic in headers if topic), "")
        started_at = max(time for time, _ in headers)
        records.append((started_at, int(match["sequence"] or 0), stem, topic, kinds))

    rounds: dict[str, int] = {}
    meetings = []
    for started_at, _, stem, topic, kinds in sorted(records):
        topic_key = " ".join(topic.casefold().split())
        number = rounds.get(topic_key, 0) + 1
        if topic_key:
            rounds[topic_key] = number
        meetings.append(
            ArchivedMeeting(
                stem,
                topic,
                started_at,
                number,
                MINUTES in kinds,
                TRANSCRIPT in kinds,
                VERBATIM in kinds,
            )
        )
    return tuple(reversed(meetings))


def read_meeting(output_dir: Path, meeting: ArchivedMeeting) -> tuple[Export, ...]:
    """Read the available minutes, edited transcript, and verbatim transcript."""
    if output_dir.is_symlink() or not re.fullmatch(STEM, meeting.stem):
        return ()
    exports = []
    for kind in EXPORT_KINDS:
        path = output_dir / f"{meeting.stem}_{kind}"
        try:
            with _open_export(path) as stream:
                exports.append(Export(path, stream.read()))
        except (OSError, UnicodeError):
            continue
    return tuple(exports)
