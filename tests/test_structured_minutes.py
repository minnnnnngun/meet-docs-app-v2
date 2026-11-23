import json
import unittest
from unittest.mock import patch

from google.genai.types import Candidate, Content, GenerateContentResponse, Part

from llm_summarizer import DecisionEvidence, GeminiSummarizer, SummaryResponseError


class StructuredMinutesTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("llm_summarizer.create_gemini_client")
        self.client = patcher.start().return_value
        self.addCleanup(patcher.stop)
        self.summarizer = GeminiSummarizer("demo-key", "정산")

    def response(
        self, *, agenda=None, decisions=None, tasks=None, finish_reason="STOP"
    ):
        result = {
            "agenda": agenda or [],
            "decisions": decisions or [],
            "action_items": tasks or [],
        }
        self.client.models.generate_content.return_value = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(parts=[Part(text=json.dumps(result))]),
                    finish_reason=finish_reason,
                )
            ]
        )

    def test_schema_and_validated_markdown_use_raw_speech(self):
        original = "[00:00:02] 구간 내 화자 1 · 김 담당자님이 낼까지 하기로 했습니다"
        self.response(
            agenda=[
                {
                    "title": "정산",
                    "summary": "결과 검토",
                    "source_line": 1,
                    "quote": "낼까지 하기로 했습니다",
                }
            ],
            decisions=[
                {
                    "status": "확정",
                    "text": "정산 검토 진행",
                    "quote": "낼까지 하기로 했습니다",
                    "source_line": 1,
                }
            ],
            tasks=[
                {
                    "task": "정산 검토",
                    "owner": "김 담당자님",
                    "due": "낼까지",
                    "quote": "낼까지 하기로 했습니다",
                    "source_line": 1,
                }
            ],
        )
        output = self.summarizer.update("", original)
        self.assertIn("## 핵심 안건\n- **정산** — 결과 검토", output)
        self.assertIn("- **확정** — 정산 검토 진행", output)
        self.assertIn("- [ ] 정산 검토 | 담당자: 김 담당자님 | 기한: 낼까지", output)
        self.assertNotIn("낼까지 하기로 했습니다", output)
        self.assertEqual(
            self.summarizer.latest_evidence,
            (DecisionEvidence("정산 검토 진행", "확정", 1, "낼까지 하기로 했습니다"),),
        )
        call = self.client.models.generate_content.call_args.kwargs
        self.assertEqual(json.loads(call["contents"])["원문 발언"], [original])
        self.assertEqual(call["config"].response_mime_type, "application/json")
        self.assertEqual(
            set(call["config"].response_schema.required),
            {"agenda", "decisions", "action_items"},
        )

    def test_uncertain_unsupported_and_speaker_label_claims_stay_unconfirmed(self):
        original = (
            "[00:00:02] 구간 내 화자 1 · [확인 필요] 금요일에 진행\n"
            "[00:00:02] 구간 내 화자 2 · 다음 주 논의"
        )
        self.response(
            decisions=[
                {
                    "status": "확정",
                    "text": "금요일 진행",
                    "quote": "금요일에 진행",
                    "source_line": 1,
                },
                {
                    "status": "확정",
                    "text": "내일 출시",
                    "quote": "내일 출시",
                    "source_line": 2,
                },
                {
                    "status": "검토",
                    "text": "일정 논의",
                    "quote": "다음 주 논의",
                    "source_line": 2,
                },
            ],
            tasks=[
                {
                    "task": "후속 일정 확인",
                    "owner": "구간 내 화자 2",
                    "due": "미정",
                    "quote": "다음 주 논의",
                    "source_line": 2,
                },
                {
                    "task": "문서 업데이트",
                    "owner": "김민수",
                    "due": "미정",
                    "quote": "다음 주 논의",
                    "source_line": 2,
                },
            ],
        )
        output = self.summarizer.update("", original)
        self.assertIn("- **확인 필요** — 금요일 진행", output)
        self.assertIn("- **확인 필요** — 내일 출시", output)
        self.assertIn("- **검토** — 일정 논의", output)
        self.assertIn("후속 일정 확인 (담당자 또는 기한 확인)", output)
        self.assertIn("문서 업데이트 (할 일 배정 근거 확인)", output)
        self.assertNotIn("- **확정**", output)
        self.assertNotIn("- [ ]", output)
        self.assertEqual(
            self.summarizer.latest_evidence[:3],
            (
                DecisionEvidence("금요일 진행", "확인 필요", None, ""),
                DecisionEvidence("내일 출시", "확인 필요", None, ""),
                DecisionEvidence("일정 논의", "검토", 2, "다음 주 논의"),
            ),
        )

    def test_prior_draft_claim_must_match_exactly(self):
        prior = "## 주요 논의 및 결정 사항\n- **확정** — 다음 주 점검"
        self.response(
            decisions=[
                {
                    "status": "확정",
                    "text": "다음 주 점검",
                    "quote": "",
                    "source_line": 0,
                },
                {
                    "status": "확정",
                    "text": "이번 주 점검",
                    "quote": "",
                    "source_line": 0,
                },
            ],
        )
        draft = self.summarizer.update(prior, "[00:00:02] 다음 안건으로 넘어가요")
        self.assertEqual(
            self.summarizer.latest_evidence[0],
            DecisionEvidence("다음 주 점검", "확정", None, ""),
        )
        final = self.summarizer.update("", "[00:00:02] 다음 안건으로 넘어가요")
        self.assertIn("- **확정** — 다음 주 점검", draft)
        self.assertIn("- **확인 필요** — 이번 주 점검", draft)
        self.assertNotIn("- **확정**", final)

    def test_evidence_requires_exact_quote_on_the_annotated_source_line(self):
        original = (
            "[00:00:02] 구간 내 화자 1 · 첫 안 검토\n"
            "[00:00:02] 구간 내 화자 2 · 둘째 안에 동의\n"
            "안건 확인했어요"
        )
        self.response(
            decisions=[
                {
                    "status": "확정",
                    "text": "잘못된 발언 참조",
                    "quote": "둘째 안에 동의",
                    "source_line": 1,
                },
                {
                    "status": "확정",
                    "text": "표시가 없는 발언",
                    "quote": "안건 확인했어요",
                    "source_line": 3,
                },
                {
                    "status": "확정",
                    "text": "둘째 안 채택",
                    "quote": "둘째 안에 동의",
                    "source_line": 2,
                },
            ]
        )
        self.summarizer.update("", original)
        self.assertEqual(
            self.summarizer.latest_evidence,
            (
                DecisionEvidence("잘못된 발언 참조", "확인 필요", None, ""),
                DecisionEvidence("표시가 없는 발언", "확인 필요", None, ""),
                DecisionEvidence("둘째 안 채택", "확정", 2, "둘째 안에 동의"),
            ),
        )
        snapshot = self.summarizer.latest_evidence
        self.response(finish_reason="MAX_TOKENS")
        with self.assertRaises(SummaryResponseError):
            self.summarizer.update("", original)
        self.assertIs(self.summarizer.latest_evidence, snapshot)

    def test_exact_proposal_quote_does_not_turn_into_an_automatic_decision(self):
        self.response(
            decisions=[
                {
                    "status": "확정",
                    "text": "A안 채택",
                    "quote": "A안으로 가시죠",
                    "source_line": 1,
                }
            ]
        )
        minutes = self.summarizer.update(
            "", "[00:00:02] 구간 내 화자 1 · A안으로 가시죠"
        )
        self.assertIn("- **확인 필요** — A안 채택", minutes)
        self.assertEqual(
            self.summarizer.latest_evidence,
            (DecisionEvidence("A안 채택", "확인 필요", 1, "A안으로 가시죠"),),
        )

    def test_rejects_unsafe_and_incomplete_payloads(self):
        for summary in (
            "<script>text</script>",
            "[열기](https://example.com)",
            "demo-key",
            "안건\n## 위장",
        ):
            with self.subTest(summary=summary):
                self.response(agenda=[{"title": "제목", "summary": summary}])
                with self.assertRaises(SummaryResponseError):
                    self.summarizer.update("", "[00:00:02] 안건입니다")
        self.response(finish_reason="MAX_TOKENS")
        with self.assertRaises(SummaryResponseError):
            self.summarizer.update("", "[00:00:02] 안건입니다")
