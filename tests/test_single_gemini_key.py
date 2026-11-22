import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np

from audio_processor import SAMPLE_RATE, AudioChunk
from llm_summarizer import DEFAULT_GEMINI_MODEL
from meeting_session import MeetingDesk


class SingleGeminiKeyTest(unittest.TestCase):
    def test_one_key_reaches_both_workers_and_writes_exports(self):
        recognized = Event()
        minutes = (
            "## 핵심 안건\n- 안건\n"
            "## 주요 논의 및 결정 사항\n- 확인된 내용 없음\n"
            "## 담당자별 할 일 (Action Items)\n- 확인된 내용 없음"
        )

        def capture(stop, emit, _warn, ready):
            ready()
            emit(AudioChunk(0, np.full(SAMPLE_RATE * 5, 900, dtype=np.int16)))
            emit(AudioChunk(5, np.full(SAMPLE_RATE * 5, 900, dtype=np.int16)))
            emit(AudioChunk(15, np.full(SAMPLE_RATE, 900, dtype=np.int16)))
            stop.wait(4)

        with (
            TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
            patch("meeting_session.capture_audio", side_effect=capture),
            patch("meeting_session.GeminiTranscriber") as transcriber,
            patch("meeting_session.GeminiSummarizer") as summarizer,
        ):

            def transcribe(chunk, *, previous_text):
                if chunk.start_seconds == 0:
                    self.assertEqual(previous_text, "")
                    return "회의 일정 확인"
                if chunk.start_seconds == 5:
                    self.assertEqual(previous_text, "회의 일정 확인")
                    return "담당자 확인"
                self.assertEqual(previous_text, "")
                recognized.set()
                return "다음 회의 논의"

            transcriber.return_value.transcribe.side_effect = transcribe
            summarizer.return_value.update.return_value = minutes
            desk = MeetingDesk()
            desk.start(
                gemini_key="fake-gemini-key",
                topic="일정 점검",
                terms="사내 용어",
                model=DEFAULT_GEMINI_MODEL,
                output_dir=Path(directory) / "output",
            )
            meeting = desk.current()
            assert meeting is not None
            try:
                self.assertTrue(recognized.wait(4))
            finally:
                meeting.request_stop()
            deadline = monotonic() + 4
            snapshot = meeting.snapshot()
            while snapshot.status != "finished" and monotonic() < deadline:
                sleep(0.02)
                snapshot = meeting.snapshot()

            transcriber.assert_called_once_with(
                "fake-gemini-key", "회의 주제: 일정 점검\n전문 용어: 사내 용어"
            )
            summarizer.assert_called_once_with(
                "fake-gemini-key",
                "일정 점검",
                model=DEFAULT_GEMINI_MODEL,
            )
            summarizer.return_value.update.assert_called_once_with(
                "",
                "[00:00:00] 회의 일정 확인\n[00:00:05] 담당자 확인\n[00:00:15] 다음 회의 논의",
            )
            self.assertTrue(snapshot.saved)
            self.assertIn(minutes, snapshot.exports[0].content)
            self.assertIn("[00:00:00] 회의 일정 확인", snapshot.exports[1].content)
