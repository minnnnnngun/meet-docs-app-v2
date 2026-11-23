import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from correction_memory import correction_hints, remember_correction, suggestions


class CorrectionMemoryTest(unittest.TestCase):
    def test_learns_only_the_changed_term_and_acronym(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            self.assertEqual(correction_hints(directory), ())
            remember_correction(
                directory,
                "오늘 김팀장 보고입니다.",
                "오늘 진팀장 보고입니다!",
                source="meeting:1",
            )
            remember_correction(
                directory,
                "내일 에이피 아이 확인합니다.",
                "내일 API 확인합니다.",
                source="meeting:2",
            )
            self.assertEqual(
                set(correction_hints(directory)),
                {("김팀장", "진팀장"), ("에이피 아이", "API")},
            )
            if os.name == "posix":
                (database,) = directory.glob(".meet-docs-corrections/*.sqlite3")
                self.assertEqual(stat.S_IMODE(database.stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(database.parent.stat().st_mode), 0o700)

    def test_conflicting_sources_are_withheld_and_a_reedit_replaces_old_learning(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            remember_correction(directory, "김팀장", "진팀장", source="first")
            remember_correction(directory, "김팀장", "진팀장", source="first")
            remember_correction(directory, "김팀장", "임팀장", source="second")
            self.assertEqual(correction_hints(directory), ())
            remember_correction(directory, "김팀장", "임팀장", source="first")
            self.assertEqual(correction_hints(directory), (("김팀장", "임팀장"),))
            remember_correction(directory, "김팀장", "김팀장", source="first")
            remember_correction(directory, "김팀장", "김팀장", source="second")
            self.assertEqual(correction_hints(directory), ())

    def test_ignores_broad_edits_separate_replacements_and_non_acronym_rewrites(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            for original, corrected in (
                ("다른 말을 했어요", "텐서 처리를 확인합니다"),
                ("가이드 작성은 김팀장 엑셀 정리", "가이드 작성은 진팀장 문서 정리"),
                ("김팀장 그리고 기능 요청", "진팀장 그리고 개선 문의"),
                ("자료 확인을 검토 다시 하세요", "자료 확인을 API 하세요"),
                ("요청 검토", "요청 검토?"),
            ):
                remember_correction(directory, original, corrected, source="meeting:1")
            self.assertEqual(correction_hints(directory), ())
            self.assertEqual(list(directory.iterdir()), [])

    def test_limits_number_of_hints(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            for i in range(30):
                remember_correction(directory, f"정산{i}", f"기획{i}", source=str(i))
            self.assertEqual(len(correction_hints(directory)), 24)

    @unittest.skipUnless(os.name == "posix", "Requires reliable symlink support")
    def test_rejects_database_symlinks_without_touching_their_target(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            target = directory / "do-not-write"
            target.write_bytes(b"untouched")
            folder = directory / ".meet-docs-corrections"
            folder.mkdir(mode=0o700)
            (folder / "pairs.sqlite3").symlink_to(target)
            with self.assertRaises(OSError):
                correction_hints(directory)
            with self.assertRaises(OSError):
                remember_correction(directory, "김팀장", "진팀장", source="meeting:1")
            self.assertEqual(target.read_bytes(), b"untouched")

    def test_suggests_close_glossary_terms_without_changing_the_original(self):
        utterance = "구버네티스 배포를 검토하고 구버네티스 점검을 진행합니다."
        self.assertEqual(
            suggestions(utterance, ["쿠버네티스", "파이프라인"]),
            (("구버네티스", "쿠버네티스"),),
        )
        self.assertEqual(suggestions(utterance, ["쿠버네티스", "구버네티스"]), ())
        self.assertEqual(suggestions("보상", ["보장"]), ())
        self.assertEqual(suggestions("구버네티스", ["투버네티스", "쿠버네티스"]), ())
        self.assertEqual(suggestions("무관한 문장입니다.", ["쿠버네티스"]), ())


if __name__ == "__main__":
    unittest.main()
