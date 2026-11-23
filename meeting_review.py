"""Store review decisions independently of original meeting exports."""

import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from meeting_history import STEM
from meeting_storage import Export, save_exports

REVIEW_SUFFIX = "검토.json"
DECISION_STATUSES = {"확정", "검토", "확인 필요"}


@dataclass(frozen=True)
class ReviewTurn:
    line: int
    time: str
    speaker: str
    original: str
    corrected: str
    intent: str
    unclear: bool


@dataclass(frozen=True)
class ReviewDecision:
    text: str
    status: str
    source_line: int | None
    quote: str


@dataclass(frozen=True)
class ReviewRecord:
    turns: tuple[ReviewTurn, ...]
    decisions: tuple[ReviewDecision, ...]
    confirmations: dict[int, str] = field(default_factory=dict)
    approvals: dict[int, int] = field(default_factory=dict)


def _review_path(output_dir: Path, stem: str) -> Path:
    if re.fullmatch(STEM, stem) is None:
        raise ValueError("Invalid meeting stem")
    if output_dir.is_symlink():
        raise OSError("Meeting output must be a directory")
    return output_dir / f"{stem}_{REVIEW_SUFFIX}"


def _validated(record: ReviewRecord) -> ReviewRecord:
    turns = {}
    for turn in record.turns:
        if (
            type(turn.line) is not int
            or turn.line <= 0
            or turn.line in turns
            or type(turn.unclear) is not bool
            or not all(
                isinstance(value, str)
                for value in (
                    turn.time,
                    turn.speaker,
                    turn.original,
                    turn.corrected,
                    turn.intent,
                )
            )
            or not turn.original.strip()
        ):
            raise ValueError("Invalid transcript turn")
        turns[turn.line] = turn

    decisions = []
    for decision in record.decisions:
        if (
            not isinstance(decision.text, str)
            or not decision.text.strip()
            or not isinstance(decision.quote, str)
            or decision.status not in DECISION_STATUSES
            or (
                decision.source_line is not None
                and type(decision.source_line) is not int
            )
        ):
            raise ValueError("Invalid meeting decision")
        source = (
            turns.get(decision.source_line)
            if decision.source_line is not None
            else None
        )
        if (
            source is None
            or source.unclear
            or "[불명확]" in source.original
            or "[확인 필요]" in source.original
            or not decision.quote.strip()
            or decision.quote not in source.original
        ):
            status = "확인 필요" if decision.status == "확정" else decision.status
            decision = replace(decision, status=status, source_line=None, quote="")
        decisions.append(decision)

    for line, confirmation in record.confirmations.items():
        if (
            type(line) is not int
            or line not in turns
            or not isinstance(confirmation, str)
            or not confirmation.strip()
        ):
            raise ValueError("Invalid transcript confirmation")
    for index, line in record.approvals.items():
        if (
            type(index) is not int
            or not 0 <= index < len(decisions)
            or type(line) is not int
            or line not in record.confirmations
        ):
            raise ValueError("Invalid decision approval")
    return replace(record, decisions=tuple(decisions))


def read_review(output_dir: Path, stem: str) -> ReviewRecord | None:
    path = _review_path(output_dir, stem)
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise OSError("Review must be a regular file")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, encoding="utf-8") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise OSError("Review must be a regular file")
        data = json.load(handle)
    try:
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unsupported review format")
        record = ReviewRecord(
            turns=tuple(ReviewTurn(**turn) for turn in data["turns"]),
            decisions=tuple(
                ReviewDecision(**decision) for decision in data["decisions"]
            ),
            confirmations={
                int(line): text for line, text in data.get("confirmations", {}).items()
            },
            approvals={
                int(index): line for index, line in data.get("approvals", {}).items()
            },
        )
        return _validated(record)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid meeting review") from exc


def review_export(output_dir: Path, stem: str, record: ReviewRecord) -> Export:
    path = _review_path(output_dir, stem)
    content = json.dumps(
        {"version": 1, **asdict(_validated(record))}, ensure_ascii=False
    )
    return Export(path, content + "\n")


@contextmanager
def _review_lock(path: Path) -> Iterator[None]:
    lock = path.with_name(f".{path.name}.lock")
    if lock.is_symlink():
        raise OSError("Review lock must be a regular file")
    descriptor = os.open(
        lock,
        os.O_CREAT
        | os.O_RDWR
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("Review lock must be a regular file")
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def confirm_turn(output_dir: Path, stem: str, line: int, text: str) -> None:
    if type(line) is not int:
        raise ValueError("Invalid transcript line")
    with _review_lock(_review_path(output_dir, stem)):
        record = read_review(output_dir, stem)
        confirmation = text.strip()
        if (
            record is None
            or not confirmation
            or not any(turn.line == line for turn in record.turns)
        ):
            raise ValueError("A saved transcript turn and confirmation are required")
        if record.confirmations.get(line) == confirmation:
            return
        if line in record.approvals.values():
            raise ValueError(
                "This confirmation is already evidence for an approved decision"
            )
        updated = replace(
            record, confirmations={**record.confirmations, line: confirmation}
        )
        save_exports((review_export(output_dir, stem, updated),))


def approve_decision(output_dir: Path, stem: str, index: int, line: int) -> None:
    if type(index) is not int or type(line) is not int:
        raise ValueError("Invalid decision or transcript line")
    with _review_lock(_review_path(output_dir, stem)):
        record = read_review(output_dir, stem)
        if (
            record is None
            or not 0 <= index < len(record.decisions)
            or line not in record.confirmations
        ):
            raise ValueError(
                "A saved decision and confirmed transcript turn are required"
            )
        if record.approvals.get(index) == line:
            return
        updated = replace(record, approvals={**record.approvals, index: line})
        save_exports((review_export(output_dir, stem, updated),))
