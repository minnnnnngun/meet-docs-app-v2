import os
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from audio_processor import SAMPLE_RATE, AudioChunk, DialogueTurn
from meeting_history import list_meetings
from meeting_review import (
    ReviewDecision,
    ReviewRecord,
    ReviewTurn,
    read_review,
    review_export,
)
from meeting_storage import Export, save_exports
from review_view import reviewed_download

APP = Path(__file__).resolve().parents[1] / "app.py"


class ReviewUiTest(unittest.TestCase):
    def test_new_meeting_opens_a_blank_workspace_and_saves_a_separate_round(self):
        recorded = Event()

        def capture(stop, emit, _warn, ready, *, bandpass_enabled):
            ready()
            emit(AudioChunk(0, np.full(SAMPLE_RATE, 900, dtype=np.int16)))
            recorded.set()
            stop.wait(4)

        with (
            TemporaryDirectory() as root,
            patch.dict(
                os.environ,
                {"MEET_DOCS_DATA_DIR": root, "GEMINI_API_KEY": "test-key"},
                clear=True,
            ),
            patch("meeting_session.datetime", wraps=datetime) as clock,
            patch("meeting_session.capture_audio", side_effect=capture),
            patch("meeting_session.GeminiTranscriber") as transcriber,
            patch("meeting_session.GeminiSummarizer") as summarizer,
        ):
            output = Path(root) / "output"
            previous = Export(
                output / "20261123_1500_회의록.md",
                "회의 시작 2026-11-23 15:00:00\n회의 주제 주간 점검\n\n이전 회의",
            )
            save_exports((previous,))
            clock.now.return_value = datetime(2026, 11, 23, 15, 0, 30)
            transcriber.return_value.transcribe.return_value = (
                DialogueTurn(1, "이번 회의 발언", "이번 회의 발언.", False),
            )
            summarizer.return_value.update.return_value = "## 핵심 안건\n- 이번 회의"
            summarizer.return_value.latest_evidence = ()

            app = AppTest.from_file(str(APP)).run()
            app.text_input(key="meeting_topic").input("이전 회의 설정")
            app.text_area(key="meeting_terms").input("이전 회의 전문 용어")
            app.get("button_group")[0].set_value("회의 기록").run()
            app.button(key="new_meeting_history").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.get("button_group")[0].value, "실시간 회의")
            self.assertEqual(app.text_input(key="meeting_topic").value, "")
            self.assertEqual(app.text_area(key="meeting_terms").value, "")
            self.assertEqual(len(list_meetings(output)), 1)

            app.text_input(key="meeting_topic").input("주간 점검")
            next(
                button for button in app.button if button.label == "회의 시작"
            ).click().run()
            self.assertTrue(recorded.wait(3))
            app.get("button_group")[0].set_value("회의 기록").run()
            self.assertTrue(app.button(key="new_meeting_history").proto.disabled)
            app.get("button_group")[0].set_value("실시간 회의").run()
            next(
                button for button in app.button if button.label == "회의 종료"
            ).click().run()
            deadline = monotonic() + 3
            while (
                not any(button.key == "new_meeting_saved" for button in app.button)
                and monotonic() < deadline
            ):
                sleep(0.04)
                app.run()
            self.assertFalse(app.exception)
            self.assertEqual(
                [
                    (record.stem, record.round_number)
                    for record in list_meetings(output)
                ],
                [("20261123_1500_2", 2), ("20261123_1500", 1)],
            )
            self.assertEqual(
                previous.path.read_text(encoding="utf-8"), previous.content
            )
            app.button(key="new_meeting_saved").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.text_input(key="meeting_topic").value, "")
            self.assertFalse(
                any(button.key == "new_meeting_saved" for button in app.button)
            )
            self.assertEqual(len(list_meetings(output)), 2)

    def test_history_anchors_quotes_and_separates_the_two_approvals(self):
        with (
            TemporaryDirectory() as root,
            patch.dict(os.environ, {"MEET_DOCS_DATA_DIR": root}, clear=True),
        ):
            output = Path(root) / "output"
            stem = "20261123_1500"
            original = "[00:00:02] 구간 내 화자 2 · 음 에이안으로 [불명확]\n"
            raw = Export(output / f"{stem}_전사원문.txt", original)
            minutes = Export(
                output / f"{stem}_회의록.md",
                "회의 시작 2026-11-23 15:00:00\n회의 주제 앱 검증\n\n"
                "## 주요 논의 및 결정 사항\n- **확정** — 먼저 기록하기\n- **확인 필요** — A안 채택\n",
            )
            record = ReviewRecord(
                (
                    ReviewTurn(
                        1,
                        "00:00:02",
                        "구간 내 화자 1",
                        "먼저 기록하기로 했습니다",
                        "",
                        "",
                        False,
                    ),
                    ReviewTurn(
                        2,
                        "00:00:02",
                        "구간 내 화자 2",
                        "음 에이안으로 [불명확] <tag>",
                        "",
                        "A안을 제안함",
                        True,
                    ),
                ),
                (
                    ReviewDecision("먼저 기록하기", "확정", 1, "기록하기로 했습니다"),
                    ReviewDecision("A안 채택", "확인 필요", None, ""),
                ),
            )
            save_exports((minutes, raw, review_export(output, stem, record)))
            app = AppTest.from_file(str(APP)).run()
            self.assertFalse(app.exception)
            self.assertFalse(
                any(button.label == "발언 확인 저장" for button in app.button)
            )
            self.assertTrue(app.toggle[0].value)

            next(
                widget
                for widget in app.get("button_group")
                if widget.key != "appearance"
            ).set_value("회의 기록").run()
            self.assertFalse(app.exception)
            html = "\n".join(str(element.proto.body) for element in app.get("html"))
            self.assertIn('href="#review-turn-1"', html)
            self.assertIn('id="review-turn-1"', html)
            self.assertIn('id="review-turn-2"', html)
            self.assertIn("의도 추정", html)
            self.assertIn("&lt;tag&gt;", html)
            self.assertNotIn("<tag>", html)
            self.assertFalse(any(button.label == "결정 확정" for button in app.button))

            next(
                widget for widget in app.selectbox if widget.label == "확인할 발언"
            ).select(2).run()
            next(
                widget for widget in app.text_area if widget.label == "확인할 문장"
            ).input("A안으로 가자는 제안입니다.")
            next(
                button for button in app.button if button.label == "발언 확인 저장"
            ).click().run()
            self.assertFalse(app.exception)
            after_confirmation = read_review(output, stem)
            assert after_confirmation is not None
            self.assertEqual(
                after_confirmation.confirmations, {2: "A안으로 가자는 제안입니다."}
            )
            self.assertFalse(after_confirmation.approvals)

            next(
                button for button in app.button if button.label == "결정 확정"
            ).click().run()
            self.assertFalse(app.exception)
            approved = read_review(output, stem)
            assert approved is not None
            self.assertEqual(approved.approvals, {1: 2})
            self.assertEqual(raw.path.read_text(encoding="utf-8"), original)
            html = "\n".join(str(element.proto.body) for element in app.get("html"))
            self.assertIn("사용자 확정", html)
            self.assertIn('href="#review-turn-2"', html)
            download = reviewed_download(minutes, approved)
            self.assertIn("- **사용자 확정** — A안 채택", download.content)
            self.assertIn("기록하기로 했습니다", download.content)
            self.assertNotIn("- **확인 필요** — A안 채택", download.content)
