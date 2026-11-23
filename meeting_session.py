"""Coordinate microphone capture, transcription, incremental summaries and exports."""

import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic

from audio_processor import (
    CHUNK_SECONDS,
    AudioChunk,
    CaptureError,
    GeminiTranscriber,
    capture_audio,
    recognition_windows,
    remove_overlap,
)
from llm_summarizer import GeminiSummarizer, SummaryResponseError
from meeting_history import EXPORT_KINDS, MINUTES, TRANSCRIPT, VERBATIM
from meeting_review import (
    REVIEW_SUFFIX,
    ReviewDecision,
    ReviewRecord,
    ReviewTurn,
    review_export,
)
from meeting_storage import Export, MeetingRecord, RecoveryJournal, save_exports

SUMMARY_INTERVAL_SECONDS = 30
AUDIO_QUEUE_SECONDS = 60
TRANSCRIPT_PREVIEW_LIMIT = 100
EMPTY_MINUTES = """## 핵심 안건
- 인식된 발언 없음

## 주요 논의 및 결정 사항
- 확인된 내용 없음

## 담당자별 할 일 (Action Items)
- 확인된 내용 없음
"""


def elapsed_time(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


@dataclass(frozen=True)
class Caption:
    seconds: float
    text: str
    verbatim: str | None = None
    speaker: int | None = None
    is_unclear: bool = False
    intent: str | None = None

    @property
    def speaker_label(self) -> str:
        return (
            f"구간 내 화자 {self.speaker}"
            if self.speaker is not None
            else "화자 미확인"
        )

    def display_text(self, *, raw: bool = False) -> str:
        text = self.verbatim if raw and self.verbatim is not None else self.text
        if self.is_unclear and "[불명확]" not in text:
            return f"[확인 필요] {text}"
        return text

    def line(self, *, raw: bool = False) -> str:
        speaker = f"{self.speaker_label} · " if self.verbatim is not None else ""
        return f"[{elapsed_time(self.seconds)}] {speaker}{self.display_text(raw=raw)}"

    def review_turn(self, line: int) -> ReviewTurn:
        return ReviewTurn(
            line,
            elapsed_time(self.seconds),
            self.speaker_label,
            self.verbatim or "",
            self.text,
            self.intent or "",
            self.is_unclear,
        )


def build_exports(
    output_dir: Path,
    stem: str,
    record: MeetingRecord,
    warnings: tuple[str, ...] = (),
    lost: int = 0,
    summary_warning: str = "",
) -> tuple[Export, ...]:
    header = f"회의 시작 {record.started_at:%Y-%m-%d %H:%M:%S}"
    if record.topic:
        header += f"\n회의 주제 {record.topic}"
    minutes = record.minutes
    if not minutes:
        minutes = (
            EMPTY_MINUTES
            if not record.captions
            else "회의록을 생성하지 못했습니다. 전체 녹취를 확인하세요."
        )
    pending = len(record.captions) - record.summarized_count
    if pending:
        minutes += (
            f"\n\n> 정리되지 않은 자막 {pending}개는 전체 녹취에서 확인할 수 있습니다."
        )
    if lost:
        minutes += f"\n\n> 음성 인식 누락 구간 {lost}개가 있습니다."
    if summary_warning:
        minutes += f"\n\n> {summary_warning}"
    transcript = "\n".join(record.captions) or "인식된 발언 없음"
    if warnings:
        transcript += "\n\n처리 알림\n" + "\n".join(warnings)
    if lost:
        transcript += f"\n음성 인식 누락 구간 {lost}개"
    exports = (
        Export(
            output_dir / f"{stem}_{MINUTES}",
            f"# 회의록\n\n{header}\n\n{minutes.rstrip()}\n",
        ),
        Export(output_dir / f"{stem}_{TRANSCRIPT}", f"{header}\n\n{transcript}\n"),
    )
    if record.verbatim_captions is None:
        return exports
    verbatim = "\n".join(record.verbatim_captions) or "인식된 발언 없음"
    exports = (
        *exports,
        Export(output_dir / f"{stem}_{VERBATIM}", f"{header}\n\n{verbatim}\n"),
    )
    if record.review_turns_json:
        review = ReviewRecord(
            tuple(ReviewTurn(**json.loads(turn)) for turn in record.review_turns_json),
            tuple(
                ReviewDecision(**item)
                for item in json.loads(record.review_decisions_json or "[]")
            ),
        )
        exports += (review_export(output_dir, stem, review),)
    return exports


def available_stem(output_dir: Path, started_at: datetime) -> str:
    stamp = started_at.strftime("%Y%m%d_%H%M")
    stem = stamp
    sequence = 2
    while any(
        (output_dir / f"{stem}_{suffix}").exists()
        for suffix in (*EXPORT_KINDS, REVIEW_SUFFIX)
    ):
        stem = f"{stamp}_{sequence}"
        sequence += 1
    return stem


@dataclass(frozen=True)
class Snapshot:
    status: str
    captions: tuple[Caption, ...]
    caption_count: int
    minutes: str
    summary_pending: int
    queue_size: int
    lost_chunks: int
    warnings: tuple[str, ...]
    exports: tuple[Export, ...]
    saved: bool
    review_decisions: tuple[ReviewDecision, ...] = ()
    evidence_turns: tuple[ReviewTurn, ...] = ()


def service_error(service: str, exc: Exception) -> str:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (401, 403):
        return f"{service} 인증 또는 접근에 실패했습니다. API 키와 모델 사용 권한을 확인하세요."
    if status == 404:
        return f"{service} 모델을 찾을 수 없습니다. 계정의 모델 사용 권한을 확인하세요."
    if status == 429:
        return f"{service} 호출 한도에 도달했습니다. 결제 및 사용량 설정을 확인하세요."
    if status == 400:
        return f"{service} 요청이 거절됐습니다. API 키와 모델 설정을 확인하세요."
    return f"{service} 처리에 실패했습니다. 인터넷 연결 또는 서비스 상태를 확인하세요."


class MeetingSession:
    def __init__(
        self,
        gemini_key: str,
        topic: str,
        terms: str,
        model: str,
        output_dir: Path,
        *,
        bandpass_enabled: bool = True,
        team_words: tuple[str, ...] = (),
    ):
        self.started_at = datetime.now()
        self.topic = topic
        self.output_dir = output_dir
        self._lock = Lock()
        self._stop = Event()
        self._capture_done = Event()
        self._stt_done = Event()
        self._audio: Queue[AudioChunk] = Queue(
            maxsize=AUDIO_QUEUE_SECONDS // CHUNK_SECONDS
        )
        self._status = "starting"
        self._recorded = False
        self._recording_started_at: float | None = None
        self._captions: list[Caption] = []
        self._minutes = ""
        self._review_decisions: tuple[ReviewDecision, ...] = ()
        self._summarized_count = 0
        self._lost_chunks = 0
        self._warnings: dict[str, str] = {}
        self._exports: tuple[Export, ...] = ()
        self._saved = False
        self._journal = RecoveryJournal(output_dir)
        self._bandpass_enabled = bandpass_enabled
        active_file = os.environ.get("MEET_DOCS_ACTIVE_FILE")
        self._active_file = Path(active_file) if active_file else None
        domain_prompt = "\n".join(
            part
            for part in (
                f"회의 주제: {topic}" if topic else "",
                f"전문 용어: {terms}" if terms else "",
                f"팀 단어 표기: {', '.join(team_words)}" if team_words else "",
            )
            if part
        )
        self._capture_thread = Thread(
            target=self._capture, name="meeting-microphone", daemon=True
        )
        self._stt_thread = Thread(
            target=self._transcribe,
            args=(gemini_key, domain_prompt),
            name="meeting-stt",
            daemon=True,
        )
        self._summary_thread = Thread(
            target=self._summarize_and_export,
            args=(gemini_key, model),
            name="meeting-summary",
            daemon=True,
        )

    def start(self) -> None:
        self._journal.acquire()
        try:
            self._journal.create(self.started_at, self.topic)
        except OSError:
            self._journal.release()
            raise
        try:
            if self._active_file is not None:
                self._active_file.touch()
        except OSError:
            try:
                self._journal.clear()
            finally:
                self._journal.release()
            raise
        self._capture_thread.start()
        self._stt_thread.start()
        self._summary_thread.start()

    def request_stop(self) -> None:
        with self._lock:
            if self._status in ("starting", "recording"):
                self._status = "finishing"
                self._stop.set()

    def snapshot(self) -> Snapshot:
        with self._lock:
            source_lines = dict.fromkeys(
                decision.source_line
                for decision in self._review_decisions
                if decision.source_line is not None
            )
            return Snapshot(
                status=self._status,
                captions=tuple(self._captions[-TRANSCRIPT_PREVIEW_LIMIT:]),
                caption_count=len(self._captions),
                minutes=self._minutes,
                summary_pending=len(self._captions) - self._summarized_count,
                queue_size=self._audio.qsize(),
                lost_chunks=self._lost_chunks,
                warnings=tuple(self._warnings.values()),
                exports=self._exports,
                saved=self._saved,
                review_decisions=self._review_decisions,
                evidence_turns=tuple(
                    self._captions[line - 1].review_turn(line) for line in source_lines
                ),
            )

    def _warn(self, source: str, message: str) -> None:
        with self._lock:
            self._warnings[source] = message

    def _audio_started(self) -> None:
        with self._lock:
            self._recorded = True
            self._recording_started_at = monotonic()
            if self._status == "starting":
                self._status = "recording"

    def _capture(self) -> None:
        try:
            capture_audio(
                self._stop,
                self._enqueue,
                lambda message: self._warn("audio", message),
                self._audio_started,
                bandpass_enabled=self._bandpass_enabled,
            )
        except CaptureError as exc:
            self._warn("capture", str(exc))
        except Exception:  # noqa: BLE001 - keep worker failures visible without leaking request data.
            self._warn(
                "capture",
                "마이크 녹음이 중단됐습니다. 시스템 기본 입력 장치와 마이크 권한을 확인하세요.",
            )
        finally:
            self.request_stop()
            self._capture_done.set()

    def _enqueue(self, chunk: AudioChunk) -> None:
        try:
            self._audio.put_nowait(chunk)
        except Full:
            with self._lock:
                self._lost_chunks += 1
                self._warnings["queue"] = (
                    "음성 인식이 밀려 일부 오디오 구간을 처리하지 못했습니다."
                )

    def _captured_chunks(self) -> Iterator[AudioChunk]:
        while not self._capture_done.is_set() or not self._audio.empty():
            try:
                yield self._audio.get(timeout=0.2)
            except Empty:
                continue

    def _transcribe(self, key: str, domain_prompt: str) -> None:
        transcriber = None
        previous_end = -1.0
        previous_raw = ""
        previous_corrected = ""
        try:
            transcriber = GeminiTranscriber(key, domain_prompt)
            for chunk in recognition_windows(self._captured_chunks()):
                adjacent = abs(chunk.start_seconds - previous_end) < 0.001
                try:
                    turns = transcriber.transcribe(
                        chunk, previous_text=previous_raw if adjacent else ""
                    )
                except Exception as exc:  # noqa: BLE001 - keep worker failures visible without leaking request data.
                    with self._lock:
                        self._lost_chunks += 1
                        self._warnings["stt"] = service_error("Gemini 음성 인식", exc)
                    continue
                last_raw = ""
                last_corrected = ""
                for index, turn in enumerate(turns):
                    verbatim = " ".join(turn.verbatim.split())
                    corrected = " ".join(turn.corrected.split())
                    intent = turn.intent
                    if index == 0 and chunk.overlap_samples and adjacent:
                        trimmed = remove_overlap(previous_raw, verbatim)
                        shorter = remove_overlap(previous_corrected, corrected)
                        if trimmed != verbatim and shorter == corrected:
                            shorter = trimmed
                        if trimmed != verbatim:
                            intent = None
                        verbatim, corrected = trimmed, shorter
                    if not verbatim:
                        continue
                    last_raw, last_corrected = verbatim, corrected or verbatim
                    self._record_caption(
                        Caption(
                            chunk.start_seconds,
                            last_corrected,
                            last_raw,
                            turn.speaker,
                            turn.is_unclear,
                            intent,
                        )
                    )
                previous_end = chunk.end_seconds
                previous_raw, previous_corrected = last_raw, last_corrected
        except Exception as exc:  # noqa: BLE001 - keep worker failures visible without leaking request data.
            self._warn("stt", service_error("Gemini 음성 인식", exc))
            self.request_stop()
        finally:
            try:
                if transcriber is not None:
                    transcriber.close()
            finally:
                self._stt_done.set()

    def _record_caption(self, caption: Caption) -> None:
        with self._lock:
            sequence = len(self._captions) + 1
        try:
            self._journal.append_caption(
                sequence,
                caption.line(),
                caption.line(raw=True),
                json.dumps(asdict(caption.review_turn(sequence)), ensure_ascii=False),
            )
        except OSError:
            self._warn_recovery()
        with self._lock:
            self._captions.append(caption)

    def _warn_recovery(self) -> None:
        self._warn(
            "recovery",
            "임시 복구 파일에 저장하지 못했습니다. 앱을 열어 둔 상태에서 회의를 종료하고 결과를 저장하세요.",
        )

    def _update_minutes(
        self, summarizer: GeminiSummarizer, *, final: bool = False
    ) -> bool:
        with self._lock:
            offset = 0 if final else self._summarized_count
            pending = self._captions[offset:]
            previous = "" if final else self._minutes
            old_decisions = (
                {}
                if final
                else {(row.text, row.status): row for row in self._review_decisions}
            )
            new_count = len(self._captions)
        if not pending:
            return False
        prefix = "최종 회의록을 작성하지 못했습니다. " if final else ""
        try:
            minutes = summarizer.update(
                previous, "\n".join(caption.line(raw=True) for caption in pending)
            )
        except SummaryResponseError as exc:
            self._warn("summary", prefix + str(exc))
            return True
        except Exception as exc:  # noqa: BLE001 - keep worker failures visible without leaking request data.
            self._warn("summary", prefix + service_error("Gemini 회의록 정리", exc))
            code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            return code is None or code in (408, 429, 500, 502, 503, 504)
        decisions = tuple(
            ReviewDecision(row.text, row.status, offset + row.source_line, row.quote)
            if row.source_line is not None
            else old_decisions.get(
                (row.text, row.status), ReviewDecision(row.text, row.status, None, "")
            )
            for row in summarizer.latest_evidence
        )
        try:
            self._journal.store_minutes(
                minutes,
                new_count,
                json.dumps(
                    [asdict(decision) for decision in decisions], ensure_ascii=False
                ),
            )
        except OSError:
            self._warn_recovery()
        with self._lock:
            self._minutes = minutes
            self._review_decisions = decisions
            self._summarized_count = new_count
            self._warnings.pop("summary", None)
        return False

    def _summarize_and_export(self, key: str, model: str) -> None:
        summarizer = None
        try:
            summarizer = GeminiSummarizer(key, self.topic, model=model)
            while not self._stt_done.wait(0.25):
                with self._lock:
                    if self._recording_started_at is not None:
                        break
            with self._lock:
                started_at = self._recording_started_at
            if started_at is not None:
                next_update = started_at + SUMMARY_INTERVAL_SECONDS
                while not self._stt_done.wait(max(0, next_update - monotonic())):
                    self._update_minutes(summarizer)
                    now = monotonic()
                    next_update += (
                        int(max(0, now - next_update) / SUMMARY_INTERVAL_SECONDS) + 1
                    ) * SUMMARY_INTERVAL_SECONDS
            if self._update_minutes(summarizer, final=True):
                self._update_minutes(summarizer, final=True)
        except Exception as exc:  # noqa: BLE001 - keep worker failures visible without leaking request data.
            self._warn("summary", service_error("Gemini 회의록 정리", exc))
        finally:
            self._stt_done.wait()
            self._capture_done.wait()
            try:
                if summarizer is not None:
                    summarizer.close()
            finally:
                self._finish()

    def _finish(self) -> None:
        if not self._recorded:
            try:
                self._journal.clear()
            except OSError:
                self._warn_recovery()
            self._set_terminal_status("failed")
            return

        stem = available_stem(self.output_dir, self.started_at)
        try:
            self._journal.store_stem(stem)
        except OSError:
            self._warn_recovery()

        with self._lock:
            record = MeetingRecord(
                self.started_at,
                self.topic,
                tuple(caption.line() for caption in self._captions),
                self._minutes,
                self._summarized_count,
                verbatim_captions=tuple(
                    caption.line(raw=True) for caption in self._captions
                ),
                review_turns_json=tuple(
                    json.dumps(asdict(caption.review_turn(line)), ensure_ascii=False)
                    for line, caption in enumerate(self._captions, start=1)
                ),
                review_decisions_json=json.dumps(
                    [asdict(decision) for decision in self._review_decisions],
                    ensure_ascii=False,
                ),
            )
            self._exports = build_exports(
                self.output_dir,
                stem,
                record,
                tuple(self._warnings.values()),
                self._lost_chunks,
                self._warnings.get("summary", ""),
            )
        self._save_exports()

    def retry_save(self) -> None:
        with self._lock:
            if self._status != "finished" or self._saved:
                return
            try:
                self._journal.acquire()
                if self._active_file is not None:
                    self._active_file.touch()
            except OSError:
                self._warnings["save"] = (
                    "복구 파일을 열 수 없습니다. 결과를 다운로드하거나 잠시 후 다시 시도하세요."
                )
                return
            self._status = "finishing"
        Thread(target=self._save_exports, name="meeting-export", daemon=True).start()

    def _save_exports(self) -> None:
        try:
            save_exports(self._exports)
        except OSError:
            self._warn(
                "save",
                "output 폴더에 저장하지 못했습니다. 폴더 권한이나 디스크 공간을 확인하고 다시 저장하거나 다운로드하세요.",
            )
        else:
            try:
                self._journal.clear()
            except OSError:
                self._warn(
                    "recovery",
                    "결과는 저장됐지만 임시 복구 파일을 정리하지 못했습니다.",
                )
            with self._lock:
                self._saved = True
                self._warnings.pop("save", None)
        finally:
            self._set_terminal_status("finished")

    def _set_terminal_status(self, status: str) -> None:
        with self._lock:
            if status == "failed" or self._saved:
                self._journal.release()
            if self._active_file is not None:
                try:
                    self._active_file.unlink(missing_ok=True)
                except OSError:
                    self._warnings["marker"] = (
                        "녹음 상태 표시를 지우지 못했습니다. 결과 저장 상태를 확인하세요."
                    )
            self._status = status


class MeetingDesk:
    """Share one microphone session across local browser tabs and page reloads."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._meeting: MeetingSession | None = None

    def current(self) -> MeetingSession | None:
        with self._lock:
            return self._meeting

    def start(self, **settings) -> None:
        with self._lock:
            if not self._can_replace_current():
                return
            if RecoveryJournal(settings["output_dir"]).exists():
                raise RuntimeError("Recover the previous meeting first")
            meeting = MeetingSession(**settings)
            meeting.start()
            self._meeting = meeting

    def prepare_new_meeting(self, output_dir: Path) -> bool:
        with self._lock:
            if not self._can_replace_current() or RecoveryJournal(output_dir).exists():
                return False
            self._meeting = None
            return True

    def has_recovery(self, output_dir: Path) -> bool:
        with self._lock:
            return self._can_replace_current() and RecoveryJournal(output_dir).exists()

    def _can_replace_current(self) -> bool:
        if self._meeting is None:
            return True
        view = self._meeting.snapshot()
        return view.status == "failed" or (view.status == "finished" and view.saved)

    def recover(self, output_dir: Path) -> tuple[Export, ...]:
        with self._lock:
            journal = RecoveryJournal(output_dir)
            if not self._can_replace_current() or not journal.exists():
                return ()
            journal.acquire()
            try:
                if not journal.exists():
                    return ()
                record = journal.load()
                if record is None:
                    journal.clear()
                    return ()
                stem = record.stem or available_stem(output_dir, record.started_at)
                journal.store_stem(stem)
                exports = build_exports(
                    output_dir,
                    stem,
                    record,
                    ("앱 중단 전에 저장된 자막과 회의록을 복구했습니다.",),
                )
                save_exports(exports)
                journal.clear()
                return exports
            finally:
                journal.release()
