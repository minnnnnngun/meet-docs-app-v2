import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np

from audio_processor import SAMPLE_RATE, AudioChunk, DialogueTurn
from llm_summarizer import DEFAULT_GEMINI_MODEL
from meeting_session import MeetingDesk
from team_glossary import read_glossary, save_glossary


class SingleGeminiKeyTest(unittest.TestCase):
    def test_one_key_reaches_both_workers_and_writes_exports(self):
        recognized = Event()
        draft_available = Event()
        minutes = (
            "## 핵심 안건\n- 안건\n"
            "## 주요 논의 및 결정 사항\n- 확인된 내용 없음\n"
            "## 담당자별 할 일 (Action Items)\n- 확인된 내용 없음"
        )

        def capture(stop, emit, _warn, ready, *, bandpass_enabled):
            self.assertFalse(bandpass_enabled)
            ready()
            active = np.full(SAMPLE_RATE * 2, 900, dtype=np.int16)
            has_pause = active.copy()
            has_pause[-SAMPLE_RATE // 4 :] = 0
            emit(AudioChunk(0, active))
            emit(AudioChunk(2, has_pause))
            emit(AudioChunk(4, has_pause))
            if not draft_available.wait(4):
                return
            emit(AudioChunk(15, np.full(SAMPLE_RATE, 900, dtype=np.int16)))
            stop.wait(4)

        with (
            TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
            patch("meeting_session.capture_audio", side_effect=capture),
            patch("meeting_session.SUMMARY_INTERVAL_SECONDS", 0.1),
            patch("meeting_session.GeminiTranscriber") as transcriber,
            patch("meeting_session.GeminiSummarizer") as summarizer,
        ):

            def transcribe(chunk, *, previous_text):
                if chunk.start_seconds == 0:
                    self.assertEqual(chunk.end_seconds, 4)
                    self.assertEqual(previous_text, "")
                    return (
                        DialogueTurn(1, "어 회의 일정 확인", "회의 일정 확인.", False),
                    )
                if chunk.start_seconds == 4:
                    self.assertEqual(previous_text, "어 회의 일정 확인")
                    return (DialogueTurn(2, "담당자 확인", "담당자 확인.", True),)
                self.assertEqual(previous_text, "")
                recognized.set()
                return (
                    DialogueTurn(None, "음 다음 회의 논의", "다음 회의 논의.", False),
                )

            def summarize(_previous, _original):
                draft_available.set()
                return minutes

            transcriber.return_value.transcribe.side_effect = transcribe
            summarizer.return_value.update.side_effect = summarize
            glossary_dir = Path(directory)
            glossary = {
                "큐플로우": "비공개 이름과 소속",
                "팀배포": "비공개 프로젝트 설명",
            }
            save_glossary(glossary_dir, json.dumps(glossary))
            loaded = read_glossary(glossary_dir)
            self.assertEqual(loaded, glossary)
            desk = MeetingDesk()
            desk.start(
                gemini_key="fake-gemini-key",
                topic="일정 점검",
                terms="사내 용어",
                model=DEFAULT_GEMINI_MODEL,
                output_dir=Path(directory) / "output",
                bandpass_enabled=False,
                team_words=tuple(loaded),
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
                "fake-gemini-key",
                "회의 주제: 일정 점검\n전문 용어: 사내 용어\n팀 단어 표기: 큐플로우, 팀배포",
            )
            summarizer.assert_called_once_with(
                "fake-gemini-key",
                "일정 점검",
                model=DEFAULT_GEMINI_MODEL,
            )
            updates = summarizer.return_value.update.call_args_list
            self.assertGreaterEqual(len(updates), 2)
            self.assertNotIn("[00:00:15]", updates[0].args[1])
            self.assertEqual(
                updates[-1].args,
                (
                    "",
                    (
                        "[00:00:00] 구간 내 화자 1 · 어 회의 일정 확인\n"
                        "[00:00:04] 구간 내 화자 2 · [확인 필요] 담당자 확인\n"
                        "[00:00:15] 화자 미확인 · 음 다음 회의 논의"
                    ),
                ),
            )
            self.assertTrue(snapshot.saved)
            self.assertIn(minutes, snapshot.exports[0].content)
            self.assertIn("· 회의 일정 확인.", snapshot.exports[1].content)
            self.assertNotIn("어 회의 일정 확인", snapshot.exports[1].content)
            self.assertIn("어 회의 일정 확인", snapshot.exports[2].content)
            self.assertTrue(snapshot.exports[2].path.name.endswith("_전사원문.txt"))
