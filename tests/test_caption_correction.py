import json
import unittest
from io import BytesIO
from unittest.mock import patch
from wave import open as open_wave

import numpy as np
from google.genai.types import Candidate, Content, GenerateContentResponse, Part

from audio_processor import (
    CONTEXT_CHARACTERS,
    FRAME_SAMPLES,
    OVERLAP_SAMPLES,
    SAMPLE_RATE,
    AudioChunk,
    DialogueTurn,
    GeminiTranscriber,
    recognition_windows,
)


class CaptionCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.patch = patch("audio_processor.create_gemini_client")
        self.client = self.patch.start().return_value
        self.addCleanup(self.patch.stop)
        self.transcriber = GeminiTranscriber("fake-api-key", "정산 배치")
        self.chunk = AudioChunk(0, np.full(SAMPLE_RATE, 900, dtype=np.int16))

    def set_response(
        self, transcript, corrected, *, has_speech=True, unclear_words=None, intent=""
    ):
        self.response_dialogues(
            [
                {
                    "speaker": 1,
                    "verbatim_text": transcript,
                    "corrected_text": corrected,
                    "intent_text": intent,
                    "is_unclear": False,
                    "unclear_words": unclear_words or [],
                }
            ],
            has_speech=has_speech,
        )

    def response_dialogues(self, dialogues, *, has_speech=True):
        payload = json.dumps({"has_speech": has_speech, "dialogues": dialogues})
        self.client.models.generate_content.return_value = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(parts=[Part(text=payload)]),
                    finish_reason="STOP",
                )
            ]
        )

    def test_single_request_carries_wave_data_and_bounded_context(self):
        self.set_response(
            "음 다음주 까지 검토 해야 해요",
            "다음 주까지 검토해야 해요.",
        )
        self.assertEqual(
            self.transcriber.transcribe(
                AudioChunk(2, self.chunk.samples, OVERLAP_SAMPLES),
                previous_text="앞" * 250,
            ),
            (
                DialogueTurn(
                    1,
                    "음 다음주 까지 검토 해야 해요",
                    "다음 주까지 검토해야 해요.",
                    False,
                ),
            ),
        )
        self.client.models.generate_content.assert_called_once()
        request = self.client.models.generate_content.call_args.kwargs
        context, audio = request["contents"]
        self.assertEqual(
            json.loads(context),
            {
                "표기 참고": "정산 배치",
                "이전 자막 참고": "앞" * CONTEXT_CHARACTERS,
                "앞부분의 겹침 길이(초)": 0.8,
            },
        )
        self.assertEqual(audio.inline_data.mime_type, "audio/wav")
        with open_wave(BytesIO(audio.inline_data.data), "rb") as stream:
            self.assertEqual((stream.getframerate(), stream.getnchannels()), (16000, 1))
        self.assertEqual(
            request["config"].response_schema.required,
            ["has_speech", "dialogues"],
        )
        self.assertIn(
            "intent_text",
            request["config"].response_schema.properties["dialogues"].items.required,
        )

    def test_intent_is_independent_and_preserves_uncertainty(self):
        self.set_response(
            "음 A안으로 가죠 [불명확]",
            "A안으로 가죠 [불명확]",
            intent="A안을 제안한 것으로 보입니다.",
        )
        (turn,) = self.transcriber.transcribe(self.chunk)
        self.assertEqual(turn.verbatim, "음 A안으로 가죠 [불명확]")
        self.assertEqual(turn.corrected, "A안으로 가죠 [불명확]")
        self.assertTrue(turn.is_unclear)
        self.assertEqual(turn.intent, "A안을 제안한 것으로 보입니다.")

    def test_sends_both_sides_of_a_short_utterance_in_one_wav(self):
        first = np.zeros(SAMPLE_RATE * 2, dtype=np.int16)
        first[-2 * FRAME_SAMPLES :] = 900
        second = np.zeros(SAMPLE_RATE * 2, dtype=np.int16)
        second[: 5 * FRAME_SAMPLES] = 1100
        (joined,) = recognition_windows(
            (
                AudioChunk(0, first),
                AudioChunk(
                    2,
                    np.concatenate((first[-OVERLAP_SAMPLES:], second)),
                    OVERLAP_SAMPLES,
                ),
            )
        )
        self.set_response("이게 뭔 냄새야", "이게 뭔 냄새야?")
        self.transcriber.transcribe(joined)
        self.client.models.generate_content.assert_called_once()
        audio = self.client.models.generate_content.call_args.kwargs["contents"][1]
        with open_wave(BytesIO(audio.inline_data.data), "rb") as stream:
            samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        np.testing.assert_array_equal(samples, np.concatenate((first, second)))

    def test_discards_intent_that_changes_numbers_negation_or_contains_key(self):
        for intent in (
            "15일에는 배포하지 않아요.",
            "5일에는 배포해요.",
            "fake-api-key",
        ):
            with self.subTest(intent=intent):
                self.set_response(
                    "5일에는 배포하지 않아요", "5일에는 배포하지 않아요.", intent=intent
                )
                (turn,) = self.transcriber.transcribe(self.chunk)
                self.assertEqual(turn.verbatim, "5일에는 배포하지 않아요")
                self.assertIsNone(turn.intent)

    def test_uses_initial_caption_if_correction_changes_numbers_or_negation(self):
        for corrected in ("15일에는 배포하지 않아요.", "5일에는 배포해요."):
            with self.subTest(corrected=corrected):
                self.set_response("5일에는 배포하지 않아요", corrected)
                self.assertEqual(
                    self.transcriber.transcribe(self.chunk),
                    (
                        DialogueTurn(
                            1,
                            "5일에는 배포하지 않아요",
                            "5일에는 배포하지 않아요",
                            False,
                        ),
                    ),
                )

    def test_keeps_existing_speech_and_credential_handling(self):
        self.set_response("", "들리지 않은 발언")
        self.assertEqual(self.transcriber.transcribe(self.chunk), ())
        self.set_response("검토합니다", "fake-api-key")
        self.assertEqual(
            self.transcriber.transcribe(self.chunk),
            (DialogueTurn(1, "검토합니다", "검토합니다", False),),
        )
        silent = AudioChunk(0, np.zeros(SAMPLE_RATE, dtype=np.int16))
        self.client.models.generate_content.reset_mock()
        self.assertEqual(self.transcriber.transcribe(silent), ())
        self.client.models.generate_content.assert_not_called()

    def test_ignores_transcript_when_gemini_reports_no_speech(self):
        self.set_response("생성된 문장", "생성된 문장.", has_speech=False)
        self.assertEqual(self.transcriber.transcribe(self.chunk), ())
        self.client.models.generate_content.assert_called_once()

    def test_marks_uncertain_terms_without_replacing_word_substrings(self):
        self.set_response(
            "백로그와 백로그의 상태 확인 [불명확]",
            "백로그와 백로그의 상태 확인. [불명확]",
            unclear_words=["백로그와"],
        )
        self.assertEqual(
            self.transcriber.transcribe(self.chunk),
            (
                DialogueTurn(
                    1,
                    "[불명확] 백로그의 상태 확인 [불명확]",
                    "[불명확] 백로그의 상태 확인. [불명확]",
                    True,
                ),
            ),
        )
        self.set_response("다음 [불명확] 확인", "다음 항목 확인")
        self.assertEqual(
            self.transcriber.transcribe(self.chunk),
            (DialogueTurn(1, "다음 [불명확] 확인", "다음 [불명확] 확인", True),),
        )

    def test_ignores_impulses_in_fresh_audio_even_after_loud_overlap(self):
        samples = np.zeros(OVERLAP_SAMPLES + SAMPLE_RATE * 2, dtype=np.int16)
        samples[:OVERLAP_SAMPLES] = 900
        samples[
            OVERLAP_SAMPLES + FRAME_SAMPLES - 1 : OVERLAP_SAMPLES + FRAME_SAMPLES + 1
        ] = 30_000
        chunk = AudioChunk(2, samples, OVERLAP_SAMPLES)
        self.assertEqual(self.transcriber.transcribe(chunk), ())
        self.client.models.generate_content.assert_not_called()

    def test_retains_quiet_sustained_audio_in_a_short_final_chunk(self):
        self.set_response("네", "네.")
        final_chunk = AudioChunk(4, np.full(FRAME_SAMPLES * 4, 80, dtype=np.int16))
        self.assertEqual(
            self.transcriber.transcribe(final_chunk),
            (DialogueTurn(1, "네", "네.", False),),
        )
        audio = self.client.models.generate_content.call_args.kwargs["contents"][1]
        with open_wave(BytesIO(audio.inline_data.data), "rb") as stream:
            samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        np.testing.assert_array_equal(samples, np.full(FRAME_SAMPLES * 4, 240))

    def test_order_and_unknown_labels_are_local_to_each_response(self):
        turn = {
            "verbatim_text": "네",
            "corrected_text": "네.",
            "is_unclear": False,
            "unclear_words": [],
        }
        self.response_dialogues(
            [
                dict(turn, speaker=1),
                dict(turn, speaker=2, verbatim_text="아 네"),
                dict(turn, speaker=0, is_unclear=True),
            ]
        )
        self.assertEqual(
            self.transcriber.transcribe(self.chunk),
            (
                DialogueTurn(1, "네", "네.", False),
                DialogueTurn(2, "아 네", "네.", False),
                DialogueTurn(None, "네 [불명확]", "네 [불명확]", True),
            ),
        )
        self.response_dialogues([dict(turn, speaker=0)])
        self.assertIsNone(self.transcriber.transcribe(self.chunk)[0].speaker)

    def test_rejects_invalid_speaker_numbers_and_real_names(self):
        turn = {
            "verbatim_text": "네",
            "corrected_text": "네.",
            "is_unclear": False,
            "unclear_words": [],
        }
        for speaker in (True, -1, 9, "김팀장"):
            with self.subTest(speaker=speaker):
                self.response_dialogues([dict(turn, speaker=speaker)])
                with self.assertRaisesRegex(TypeError, "발언 형식"):
                    self.transcriber.transcribe(self.chunk)
