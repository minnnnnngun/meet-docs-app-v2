"""Apply confirmed text and recheck only the audio surrounding a selected turn."""

from collections.abc import Mapping
from pathlib import Path

from audio_processor import GeminiTranscriber, RecheckTarget
from correction_memory import remember_correction
from llm_summarizer import GeminiSummarizer
from meeting_audio import read_clip
from meeting_review import (
    ReviewDecision,
    ReviewRecord,
    ReviewTurn,
    save_revised_summary,
)


def corrected_lines(
    turns: tuple[ReviewTurn, ...], corrections: Mapping[int, str]
) -> str:
    rows = []
    for turn in turns:
        corrected = corrections.get(turn.line)
        text = corrected or turn.original
        if corrected is None and turn.unclear and "[불명확]" not in text:
            text = f"[확인 필요] {text}"
        rows.append(f"[{turn.time}] {turn.speaker} · {text}")
    return "\n".join(rows)


def remember_turn(data_dir: Path, stem: str, turn: ReviewTurn, correction: str) -> None:
    remember_correction(
        data_dir, turn.original, correction, source=f"{stem}:{turn.line}"
    )


def remember_review(data_dir: Path, stem: str, record: ReviewRecord) -> None:
    for turn in record.turns:
        if turn.line in record.corrections:
            remember_turn(data_dir, stem, turn, record.corrections[turn.line])


def recheck_turn(
    output_dir: Path, audio_id: str, turn: ReviewTurn, hint: str, api_key: str
) -> str:
    hours, minutes, seconds = map(int, turn.time.split(":"))
    target = hours * 3600 + minutes * 60 + seconds
    chunk = read_clip(output_dir, audio_id, target)
    transcriber = GeminiTranscriber(api_key, "")
    try:
        turns = transcriber.transcribe(
            chunk,
            target=RecheckTarget(
                target - chunk.start_seconds, turn.original, hint.strip()
            ),
        )
    finally:
        transcriber.close()
    if len(turns) != 1:
        raise ValueError("선택한 발언을 하나로 구분하지 못했습니다.")
    return turns[0].corrected


def regenerate_saved_minutes(
    output_dir: Path,
    stem: str,
    topic: str,
    record: ReviewRecord,
    api_key: str,
    model: str,
) -> ReviewRecord:
    if not record.turns:
        raise ValueError("다시 정리할 전사 기록이 없습니다.")
    summary = GeminiSummarizer(api_key, topic, model=model)
    try:
        minutes = summary.update(
            "",
            corrected_lines(record.turns, record.corrections),
            unedited_transcript=corrected_lines(record.turns, {}),
        )
        decisions = tuple(
            ReviewDecision(
                row.text,
                row.status,
                record.turns[row.source_line - 1].line if row.source_line else None,
                row.quote,
            )
            for row in summary.latest_evidence
        )
    finally:
        summary.close()
    return save_revised_summary(output_dir, stem, record, minutes, decisions)
