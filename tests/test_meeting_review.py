import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from meeting_review import (
    ReviewDecision,
    ReviewRecord,
    ReviewTurn,
    approve_decision,
    confirm_turn,
    read_review,
    review_export,
)
from meeting_storage import RecoveryJournal, save_exports

STEM = "20260102_0304"
TURN = ReviewTurn(
    1,
    "00:00:02",
    "구간 내 화자 1",
    "음 A안으로 가시죠",
    "A안으로 가시죠",
    "A안을 제안",
    False,
)
UNCLEAR = ReviewTurn(
    2, "00:00:02", "구간 내 화자 2", "B안 [불명확]", "B안 [불명확]", "", False
)
DECISION = ReviewDecision("A안 논의", "검토", 1, "A안으로 가시죠")


class MeetingReviewTest(unittest.TestCase):
    def test_transcript_confirmation_and_decision_approval_are_separate(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            record = ReviewRecord((TURN,), (DECISION,))
            export = review_export(output, STEM, record)
            save_exports((export,))
            if os.name == "posix":
                self.assertEqual(export.path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(ValueError):
                approve_decision(output, STEM, 0, 1)
            confirm_turn(output, STEM, 1, "A안을 제안했습니다")
            confirmed = read_review(output, STEM)
            self.assertIsNotNone(confirmed)
            assert confirmed is not None
            self.assertEqual(confirmed.approvals, {})
            self.assertEqual(confirmed.turns, record.turns)
            approve_decision(output, STEM, 0, 1)
            approve_decision(output, STEM, 0, 1)
            with self.assertRaises(ValueError):
                confirm_turn(output, STEM, 1, "다른 발언")
            approved = read_review(output, STEM)
            self.assertIsNotNone(approved)
            assert approved is not None
            self.assertEqual(approved.approvals, {0: 1})
            self.assertEqual(approved.confirmations, {1: "A안을 제안했습니다"})
            self.assertEqual(approved.turns, record.turns)

    def test_loaded_quotes_are_rechecked_while_turn_ids_disambiguate_same_timestamp(
        self,
    ):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            export = review_export(
                output, STEM, ReviewRecord((TURN, UNCLEAR), (DECISION, DECISION))
            )
            data = json.loads(export.content)
            data["decisions"][1].update(status="확정", source_line=2, quote="B안")
            export.path.write_text(json.dumps(data), encoding="utf-8")
            record = read_review(output, STEM)
            assert record is not None
            self.assertEqual(record.decisions[0], DECISION)
            self.assertEqual(
                record.decisions[1], ReviewDecision("A안 논의", "확인 필요", None, "")
            )

    def test_parallel_confirmation_updates_preserve_both_turns(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            save_exports(
                (review_export(output, STEM, ReviewRecord((TURN, UNCLEAR), ())),)
            )
            with ThreadPoolExecutor(max_workers=2) as pool:
                failures = [
                    pool.submit(confirm_turn, output, STEM, line, text)
                    for line, text in [(1, "A안"), (2, "B안")]
                ]
                for task in failures:
                    task.result()
            record = read_review(output, STEM)
            assert record is not None
            self.assertEqual(record.confirmations, {1: "A안", 2: "B안"})

    def test_recovery_journal_keeps_review_data_with_transcript_and_summary(self):
        with TemporaryDirectory() as directory:
            journal = RecoveryJournal(Path(directory) / "output")
            journal.acquire()
            try:
                journal.create(datetime(2026, 1, 2, 3, 4), "검토 회의")
                journal.append_caption(1, "보정 자막", "전사 원문", '{"line":1}')
                journal.store_minutes("회고", 1, '[{"text":"확인"}]')
                record = journal.load()
                assert record is not None
                self.assertEqual(record.review_turns_json, ('{"line":1}',))
                self.assertEqual(record.review_decisions_json, '[{"text":"확인"}]')
                self.assertEqual(record.verbatim_captions, ("전사 원문",))
            finally:
                journal.release()

    def test_missing_and_unrelated_paths_do_not_open_a_review(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertIsNone(read_review(output, STEM))
            with self.assertRaises(ValueError):
                read_review(output, "../private")
            target = output / "other.txt"
            target.write_text("보호할 파일", encoding="utf-8")
            link = output / f"{STEM}_검토.json"
            try:
                link.symlink_to(target)
            except (NotImplementedError, OSError):
                self.skipTest("Symlinks unavailable")
            with self.assertRaises(OSError):
                read_review(output, STEM)
