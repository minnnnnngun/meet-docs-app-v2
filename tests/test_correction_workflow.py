import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
from google.genai.types import Candidate, Content, GenerateContentResponse, Part
from streamlit.testing.v1 import AppTest

from audio_processor import SAMPLE_RATE, AudioChunk, DialogueTurn, RecheckTarget
from correction_memory import correction_hints
from llm_summarizer import GeminiSummarizer
from meeting_audio import AudioTape, audio_available
from meeting_corrections import recheck_turn
from meeting_history import list_meetings
from meeting_review import ReviewRecord, ReviewTurn, read_review, review_export
from meeting_session import MeetingDesk
from meeting_storage import Export, save_exports

APP = Path(__file__).resolve().parents[1] / "app.py"
TURN = ReviewTurn(1, "00:00:15", "구간 내 화자 1", "김팀장이 맡습니다", "", "", False)


class CorrectionWorkflowTest(unittest.TestCase):
    def test_recheck_sends_only_the_selected_window_and_returns_a_proposal(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            tape = AudioTape(output)
            tape.append(AudioChunk(0, np.full(SAMPLE_RATE * 30, 900, dtype=np.int16)))
            tape.close()
            with patch("meeting_corrections.GeminiTranscriber") as transcriber:
                transcriber.return_value.transcribe.return_value = (
                    DialogueTurn(1, "진팀장이 맡습니다", "진팀장이 맡습니다.", False),
                )
                result = recheck_turn(output, tape.audio_id, TURN, "진팀장", "fake-key")
            request = transcriber.return_value.transcribe.call_args
            clip = request.args[0]
            self.assertEqual((clip.start_seconds, clip.end_seconds), (5, 25))
            self.assertEqual(
                request.kwargs["target"], RecheckTarget(10, TURN.original, "진팀장")
            )
            self.assertEqual(result, "진팀장이 맡습니다.")

    def test_corrected_text_updates_tasks_and_preserves_original_quote_authority(self):
        with patch("llm_summarizer.create_gemini_client") as client:
            reply = {
                "agenda": [],
                "decisions": [
                    {
                        "source_line": 1,
                        "quote": "진팀장이 맡기로 했습니다",
                        "status": "확정",
                        "text": "진팀장 담당",
                    }
                ],
                "action_items": [
                    {
                        "source_line": 1,
                        "quote": "진팀장이 맡기로 했습니다",
                        "task": "문서 검토",
                        "owner": "진팀장",
                        "due": "미정",
                    }
                ],
            }
            client.return_value.models.generate_content.return_value = (
                GenerateContentResponse(
                    candidates=[
                        Candidate(
                            content=Content(parts=[Part(text=json.dumps(reply))]),
                            finish_reason="STOP",
                        )
                    ]
                )
            )
            summarizer = GeminiSummarizer("fake-key", "문서")
            result = summarizer.update(
                "",
                "[00:00:15] 구간 내 화자 1 · 문서 검토를 진팀장이 맡기로 했습니다",
                unedited_transcript="[00:00:15] 구간 내 화자 1 · 문서 검토를 김팀장이 맡기로 했습니다",
            )
        sent = client.return_value.models.generate_content.call_args.kwargs["contents"]
        self.assertIn("진팀장", sent)
        self.assertNotIn("김팀장", sent)
        self.assertIn("- **확인 필요** — 진팀장 담당", result)
        self.assertIn("담당자: 진팀장", result)
        self.assertIsNone(summarizer.latest_evidence[0].source_line)

    def test_live_edit_is_journaled_and_reused_after_saving(self):
        with (
            TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
            patch("meeting_session.capture_audio") as mic,
            patch("meeting_session.GeminiTranscriber") as transcriber,
            patch("meeting_session.GeminiSummarizer") as summarizer,
        ):

            def capture(stop, emit, _warn, ready, *, bandpass_enabled):
                ready()
                emit(AudioChunk(0, np.full(SAMPLE_RATE, 900, dtype=np.int16)))
                stop.wait(5)

            mic.side_effect = capture
            transcriber.return_value.transcribe.return_value = (
                DialogueTurn(1, TURN.original, TURN.original, False),
            )
            summarizer.return_value.latest_evidence = ()
            summarizer.return_value.update.return_value = "## 핵심 안건\n- 검토"
            desk = MeetingDesk()
            output = Path(directory) / "output"
            desk.start(
                gemini_key="fake-key",
                topic="문서",
                terms="",
                model="mock",
                output_dir=output,
                retain_audio=True,
            )
            meeting = desk.current()
            assert meeting is not None
            try:
                deadline = monotonic() + 3
                while meeting.snapshot().caption_count < 1 and monotonic() < deadline:
                    sleep(0.025)
                with (
                    patch(
                        "meeting_session.RecoveryJournal.store_corrections",
                        side_effect=OSError,
                    ),
                    self.assertRaises(OSError),
                ):
                    meeting.save_live_correction(1, "임시 교정문")
                self.assertEqual(meeting.snapshot().corrections, {})
                meeting.save_live_correction(1, "진팀장이 맡습니다")
                meeting.request_resummary()
                while (
                    not summarizer.return_value.update.called and monotonic() < deadline
                ):
                    sleep(0.025)
                call = summarizer.return_value.update.call_args
                self.assertIn("진팀장이", call.args[1])
                self.assertIn("김팀장이", call.kwargs["unedited_transcript"])
                self.assertEqual(
                    meeting.snapshot().corrections, {1: "진팀장이 맡습니다"}
                )
            finally:
                meeting.request_stop()
            deadline = monotonic() + 3
            while not meeting.snapshot().saved and monotonic() < deadline:
                sleep(0.025)
            self.assertTrue(meeting.snapshot().saved)
            stem = list_meetings(output)[0].stem
            record = read_review(output, stem)
            assert record is not None and record.audio_id is not None
            self.assertTrue(audio_available(output, record.audio_id))
            self.assertEqual(record.turns[0].original, TURN.original)
            self.assertEqual(record.corrections[1], "진팀장이 맡습니다")
            self.assertEqual(
                correction_hints(Path(directory)), (("김팀장이", "진팀장이"),)
            )

    def test_history_editor_applies_only_saved_corrections_and_refreshes_summary(self):
        with (
            TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {"MEET_DOCS_DATA_DIR": directory, "GEMINI_API_KEY": "fake-key"},
                clear=True,
            ),
            patch("meeting_corrections.GeminiSummarizer") as summarizer,
            patch(
                "correction_view.recheck_turn", return_value="진팀장이 맡습니다"
            ) as recheck,
        ):
            output = Path(directory) / "output"
            stem = "20261123_1500"
            tape = AudioTape(output)
            tape.append(AudioChunk(0, np.full(20 * SAMPLE_RATE, 900, dtype=np.int16)))
            tape.close()
            header = "회의 시작 2026-11-23 15:00:00\n회의 주제 문서\n\n"
            original = Export(output / f"{stem}_전사원문.txt", header + TURN.original)
            save_exports(
                (
                    Export(
                        output / f"{stem}_회의록.md",
                        header + "## 핵심 안건\n- 이전 회의록",
                    ),
                    original,
                    review_export(
                        output, stem, ReviewRecord((TURN,), (), audio_id=tape.audio_id)
                    ),
                )
            )
            summarizer.return_value.latest_evidence = ()
            summarizer.return_value.update.return_value = (
                "## 핵심 안건\n- 교정 반영된 회의록"
            )

            app = AppTest.from_file(str(APP)).run()
            app.get("button_group")[0].set_value("회의 기록").run()
            next(
                widget for widget in app.text_input if widget.label == "재인식 힌트"
            ).input("진팀장").run()
            next(
                button
                for button in app.button
                if button.label == "힌트 입력 후 다시 듣기"
            ).click().run()
            deadline = monotonic() + 3
            while (
                not any(button.label == "제안을 교정문에 넣기" for button in app.button)
                and monotonic() < deadline
            ):
                sleep(0.05)
                app.run()
            recheck.assert_called_once_with(
                output, tape.audio_id, TURN, "진팀장", "fake-key"
            )
            next(
                button
                for button in app.button
                if button.label == "제안을 교정문에 넣기"
            ).click().run()
            self.assertEqual(
                next(
                    widget for widget in app.text_area if widget.label == "교정문"
                ).value,
                "진팀장이 맡습니다",
            )
            self.assertEqual(correction_hints(Path(directory)), ())
            next(
                button for button in app.button if button.label == "교정문 저장"
            ).click().run()
            record = read_review(output, stem)
            assert record is not None
            self.assertEqual(record.corrections, {1: "진팀장이 맡습니다"})
            next(
                button for button in app.button if button.label == "회의록 다시 정리"
            ).click().run()

            def document() -> str:
                return "\n".join(element.proto.body for element in app.get("html"))

            deadline = monotonic() + 3
            while "교정 반영된 회의록" not in document() and monotonic() < deadline:
                sleep(0.05)
                app.run()
            self.assertFalse(app.exception)
            self.assertIn("교정 반영된 회의록", document())
            self.assertEqual(
                original.path.read_text(encoding="utf-8"), original.content
            )
