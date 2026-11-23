import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from meeting_history import list_meetings, read_meeting
from meeting_session import available_stem, build_exports
from meeting_storage import MeetingRecord, save_exports


class MeetingHistoryTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output_dir = Path(directory.name) / "output"

    def save_meeting(self, stem, started_at, topic):
        record = MeetingRecord(
            started_at,
            topic,
            ("[00:00:02] 다음 주에 확인하는 걸로 하겠습니다.",),
            "## 주요 논의\n- 다음 주에 확인합니다.",
            1,
        )
        exports = build_exports(self.output_dir, stem, record)
        save_exports(exports)
        return exports

    def test_existing_exports_and_per_topic_round_numbers(self):
        before = datetime(2026, 11, 22, 18, 30, 4)
        after = before.replace(second=49)
        earlier = self.save_meeting("20261122_1830", before, "주간 회의")
        later = self.save_meeting("20261122_1830_2", after, "주간  회의")
        weekend = self.save_meeting(
            "20261129_0900", datetime(2026, 11, 29, 9), "다른 회의"
        )

        meetings = list_meetings(self.output_dir)
        self.assertEqual(
            [meeting.stem for meeting in meetings],
            ["20261129_0900", "20261122_1830_2", "20261122_1830"],
        )
        self.assertEqual([meeting.round_number for meeting in meetings], [1, 2, 1])
        self.assertFalse(meetings[1].has_verbatim)
        self.assertEqual(meetings[1].started_at, after)
        self.assertEqual(meetings[1].topic, "주간  회의")
        self.assertEqual(read_meeting(self.output_dir, meetings[2]), earlier)
        self.assertEqual(read_meeting(self.output_dir, meetings[1]), later)
        self.assertEqual(read_meeting(self.output_dir, meetings[0]), weekend)

    def test_archives_third_export_and_verbatim_only_meetings(self):
        started_at = datetime(2026, 11, 22, 18, 30, 4)
        record = MeetingRecord(
            started_at,
            "주간 회의",
            ("[00:00:02] 다음 주에 확인합니다.",),
            "## 주요 논의\n- 다음 주에 확인합니다.",
            1,
            verbatim_captions=("[00:00:02] 다음주 어 [불명확]",),
        )
        exports = build_exports(self.output_dir, "20261122_1830", record)
        save_exports(exports)
        self.assertEqual(len(exports), 3)
        self.assertEqual(exports[2].path.name, "20261122_1830_전사원문.txt")
        meeting = list_meetings(self.output_dir)[0]
        self.assertTrue(meeting.has_verbatim)
        self.assertEqual(read_meeting(self.output_dir, meeting), exports)

        exports[0].path.unlink()
        exports[1].path.unlink()
        meeting = list_meetings(self.output_dir)[0]
        self.assertEqual(meeting.started_at, started_at)
        self.assertEqual(meeting.topic, "주간 회의")
        self.assertTrue(meeting.has_verbatim)
        self.assertFalse(meeting.has_transcript)
        self.assertEqual(read_meeting(self.output_dir, meeting), exports[2:])
        self.assertEqual(available_stem(self.output_dir, started_at), "20261122_1830_2")

    def test_partial_and_untitled_files_remain_accessible(self):
        self.assertFalse(list_meetings(self.output_dir))
        self.output_dir.mkdir()
        for suffix in ("", "_2", "_10"):
            path = self.output_dir / f"20261122_1830{suffix}_전체녹취.txt"
            path.write_text("[00:00:02] 이전 녹취.\n", encoding="utf-8-sig")
        (self.output_dir / "20261322_1830_회의록.md").write_text(
            "잘못된 날짜", encoding="utf-8"
        )
        (self.output_dir / "unrelated.txt").write_text("다른 파일", encoding="utf-8")

        meetings = list_meetings(self.output_dir)
        self.assertEqual(
            [meeting.stem for meeting in meetings],
            ["20261122_1830_10", "20261122_1830_2", "20261122_1830"],
        )
        self.assertEqual([meeting.round_number for meeting in meetings], [1, 1, 1])
        self.assertTrue(meetings[0].has_transcript)
        self.assertFalse(meetings[0].has_minutes)
        self.assertEqual(
            read_meeting(self.output_dir, meetings[0])[0].content,
            "[00:00:02] 이전 녹취.\n",
        )

    def test_rejects_symlink_and_invalid_archive_stem(self):
        exports = self.save_meeting(
            "20261122_1830", datetime(2026, 11, 22, 18, 30), "회의"
        )
        other = self.output_dir.parent / "unrelated.txt"
        other.touch()
        link = self.output_dir / "20261122_1831_회의록.md"
        try:
            link.symlink_to(other)
        except (NotImplementedError, OSError):
            self.skipTest("Symlinks unavailable")

        meetings = list_meetings(self.output_dir)
        self.assertEqual(len(meetings), 1)
        self.assertFalse(
            read_meeting(self.output_dir, replace(meetings[0], stem="../unrelated"))
        )
        exports[0].path.unlink()
        exports[0].path.symlink_to(other)
        self.assertEqual(read_meeting(self.output_dir, meetings[0]), exports[1:])


if __name__ == "__main__":
    unittest.main()
