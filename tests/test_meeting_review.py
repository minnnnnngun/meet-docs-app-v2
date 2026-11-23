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
    save_correction,
    save_revised_summary,
)
from meeting_storage import RecoveryJournal, save_exports

STEM = "20260102_0304"
AUDIO_ID = "a" * 32
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
    def test_correction_preserves_the_original_and_can_be_reverted(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            record = ReviewRecord(
                (TURN,), (DECISION,), {1: "기존 승인 근거"}, {0: 1}, audio_id=AUDIO_ID
            )
            save_exports((review_export(output, STEM, record),))
            edited = save_correction(output, STEM, 1, "  B안으로\n 가시죠  ")
            self.assertEqual(edited.corrections, {1: "B안으로 가시죠"})
            self.assertEqual(edited.turns, record.turns)
            self.assertEqual(edited.confirmations, record.confirmations)
            self.assertEqual(edited.approvals, record.approvals)
            self.assertEqual(edited.audio_id, AUDIO_ID)
            self.assertEqual(read_review(output, STEM), edited)
            with self.assertRaises(ValueError):
                save_correction(output, STEM, 1, "길" * 521)
            with self.assertRaises(ValueError):
                save_correction(output, STEM, 2, "B안")
            undone = save_correction(output, STEM, 1, TURN.original)
            self.assertEqual(undone, record)
            self.assertEqual(read_review(output, STEM), record)

    def test_revised_summary_rejects_stale_input_and_keeps_manual_approvals(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            original = ReviewRecord((TURN,), (DECISION,), {1: "기존 승인 근거"}, {0: 1})
            save_exports((review_export(output, STEM, original),))
            edited = save_correction(output, STEM, 1, "B안으로 결정했습니다")
            draft = "## 주요 논의 및 결정 사항\n- **확정** — B안"
            unproven = ReviewDecision("B안", "확정", 1, "B안으로 결정했습니다")
            with self.assertRaisesRegex(ValueError, "회의 기록이 바뀌었습니다"):
                save_revised_summary(output, STEM, original, draft, (unproven,))
            self.assertEqual(read_review(output, STEM), edited)
            revised = save_revised_summary(output, STEM, edited, draft, (unproven,))
            self.assertEqual(
                revised.decisions[0], ReviewDecision("B안", "확인 필요", None, "")
            )
            self.assertEqual(revised.decisions[1], DECISION)
            self.assertEqual(revised.approvals, {1: 1})
            self.assertEqual(revised.confirmations, original.confirmations)
            self.assertIn("- **검토** — A안 논의", revised.revised_minutes or "")
            self.assertEqual(read_review(output, STEM), revised)
            remapped = save_revised_summary(
                output,
                STEM,
                revised,
                "## 주요 논의 및 결정 사항\n- **확정** — A안 논의",
                (ReviewDecision(DECISION.text, "확정", 1, "가시죠"),),
            )
            self.assertEqual(remapped.decisions, original.decisions)
            self.assertEqual(remapped.approvals, original.approvals)

    def test_legacy_sidecar_defaults_correction_fields(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            record = ReviewRecord((TURN,), (DECISION,))
            export = review_export(output, STEM, record)
            data = json.loads(export.content)
            for field in ("corrections", "revised_minutes", "audio_id"):
                del data[field]
            export.path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(read_review(output, STEM), record)

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
                journal.store_audio_id(AUDIO_ID)
                journal.append_caption(1, "보정 자막", "전사 원문", '{"line":1}')
                journal.store_corrections({1: "교정한 전문 용어"})
                with self.assertRaises(ValueError):
                    journal.store_corrections({2: "알 수 없는 발언"})
                journal.store_minutes("회고", 1, '[{"text":"확인"}]')
                record = journal.load()
                assert record is not None
                self.assertEqual(record.review_turns_json, ('{"line":1}',))
                self.assertEqual(record.review_decisions_json, '[{"text":"확인"}]')
                self.assertEqual(record.verbatim_captions, ("전사 원문",))
                self.assertEqual(record.corrections, {1: "교정한 전문 용어"})
                self.assertEqual(record.audio_id, AUDIO_ID)
                journal.store_corrections({})
                cleared = journal.load()
                assert cleared is not None
                self.assertEqual(cleared.corrections, {})
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
