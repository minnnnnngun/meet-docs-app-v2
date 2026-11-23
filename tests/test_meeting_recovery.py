import os
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from meeting_session import MeetingDesk
from meeting_storage import RecoveryJournal, save_exports


class MeetingRecoveryTest(unittest.TestCase):
    def test_recovers_commits_after_process_exits_and_leaves_existing_export(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            existing = output / "20260102_0304_회의록.md"
            existing.write_text("기존 회의", encoding="utf-8")
            self._write_and_exit(output)
            desk = MeetingDesk()
            self.assertTrue(desk.has_recovery(output))
            exports = desk.recover(output)
            self.assertFalse(desk.has_recovery(output))
            self.assertEqual(len(exports), 2)
            self.assertEqual(existing.read_text(encoding="utf-8"), "기존 회의")
            self.assertEqual(exports[0].path.name, "20260102_0304_2_회의록.md")
            self.assertIn("정리되지 않은 자막 1개", exports[0].content)
            self.assertIn("[00:00:10] 마지막 발언", exports[1].content)
            for export in exports:
                self.assertEqual(
                    export.path.read_text(encoding="utf-8"), export.content
                )
                if os.name == "posix":
                    self.assertEqual(export.path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(desk.recover(output), ())

    def test_recovers_verbatim_separately_after_process_exits(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            self._write_and_exit(output, include_verbatim=True)
            recovered = RecoveryJournal(output).load()
            self.assertIsNotNone(recovered)
            assert recovered is not None
            self.assertEqual(
                recovered.verbatim_captions,
                ("[00:00:00] 어 첫 발언", "[00:00:10] 마지막 [불명확]"),
            )
            exports = MeetingDesk().recover(output)
            self.assertEqual(len(exports), 3)
            self.assertIn("[00:00:10] 마지막 발언", exports[1].content)
            self.assertIn("[00:00:10] 마지막 [불명확]", exports[2].content)

    def _write_and_exit(self, output: Path, *, include_verbatim: bool = False) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from meeting_storage import RecoveryJournal
journal = RecoveryJournal(Path(sys.argv[1]))
journal.acquire()
journal.create(datetime(2026, 1, 2, 3, 4), "주간 회의")
if sys.argv[2] == "verbatim":
    journal.append_caption(1, "[00:00:00] 첫 발언", "[00:00:00] 어 첫 발언")
    journal.append_caption(3, "[00:00:10] 마지막 발언", "[00:00:10] 마지막 [불명확]")
else:
    with sqlite3.connect(journal.path) as connection:
        connection.execute("ALTER TABLE captions DROP COLUMN verbatim_line")
        connection.executemany(
            "INSERT INTO captions (sequence, line) VALUES (?, ?)",
            [(1, "[00:00:00] 첫 발언"), (3, "[00:00:10] 마지막 발언")],
        )
journal.store_minutes("## 핵심 안건\\n- 첫 안건", 1)
if sys.argv[2] == "legacy":
    with sqlite3.connect(journal.path) as connection:
        connection.execute("ALTER TABLE captions DROP COLUMN review_turn_json")
        connection.execute("ALTER TABLE meeting DROP COLUMN review_decisions_json")
        connection.execute("ALTER TABLE meeting DROP COLUMN corrections_json")
        connection.execute("ALTER TABLE meeting DROP COLUMN audio_id")
os._exit(9)
""",
                str(output),
                "verbatim" if include_verbatim else "legacy",
            ],
            check=False,
        )
        self.assertEqual(result.returncode, 9)

    def test_failed_second_export_leaves_journal_and_reuses_filenames(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            self._write_and_exit(output)
            desk = MeetingDesk()

            def save_first_only(exports):
                save_exports(exports[:1])
                raise OSError("simulated disk write failure")

            with (
                patch("meeting_session.save_exports", side_effect=save_first_only),
                self.assertRaises(OSError),
            ):
                desk.recover(output)
            self.assertTrue(desk.has_recovery(output))
            self.assertEqual(len(list(output.glob("*"))), 1)
            exports = desk.recover(output)
            self.assertEqual(len(list(output.glob("*"))), 2)
            self.assertEqual(exports[0].path.name, "20260102_0304_회의록.md")
            self.assertFalse(desk.has_recovery(output))

    def test_active_meeting_cannot_be_recovered_by_another_instance(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            journal = RecoveryJournal(output)
            journal.acquire()
            try:
                journal.create(datetime(2026, 1, 2), "주간 회의")
                if os.name == "posix":
                    self.assertEqual(journal.path.stat().st_mode & 0o777, 0o600)
                with self.assertRaises(OSError):
                    MeetingDesk().recover(output)
                self.assertTrue(journal.path.exists())
            finally:
                journal.release()
