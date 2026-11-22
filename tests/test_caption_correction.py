import json
import unittest
from io import BytesIO
from unittest.mock import patch
from wave import open as open_wave

import numpy as np
from google.genai.types import Candidate, Content, GenerateContentResponse, Part

from audio_processor import (
    CONTEXT_CHARACTERS,
    SAMPLE_RATE,
    AudioChunk,
    GeminiTranscriber,
)


class CaptionCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.patch = patch("audio_processor.create_gemini_client")
        self.client = self.patch.start().return_value
        self.addCleanup(self.patch.stop)
        self.transcriber = GeminiTranscriber("fake-api-key", "정산 배치")
        self.chunk = AudioChunk(0, np.full(SAMPLE_RATE, 900, dtype=np.int16))

    def set_response(self, transcript, corrected):
        payload = json.dumps({"transcript": transcript, "corrected": corrected})
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
            self.transcriber.transcribe(self.chunk, previous_text="앞" * 250),
            "다음 주까지 검토해야 해요.",
        )
        self.client.models.generate_content.assert_called_once()
        request = self.client.models.generate_content.call_args.kwargs
        context, audio = request["contents"]
        self.assertEqual(
            json.loads(context),
            {"표기 참고": "정산 배치", "이전 자막 참고": "앞" * CONTEXT_CHARACTERS},
        )
        self.assertEqual(audio.inline_data.mime_type, "audio/wav")
        with open_wave(BytesIO(audio.inline_data.data), "rb") as stream:
            self.assertEqual((stream.getframerate(), stream.getnchannels()), (16000, 1))
        self.assertEqual(
            request["config"].response_schema.required, ["transcript", "corrected"]
        )

    def test_uses_initial_caption_if_correction_changes_numbers_or_negation(self):
        for corrected in ("15일에는 배포하지 않아요.", "5일에는 배포해요."):
            with self.subTest(corrected=corrected):
                self.set_response("5일에는 배포하지 않아요", corrected)
                self.assertEqual(
                    self.transcriber.transcribe(self.chunk), "5일에는 배포하지 않아요"
                )

    def test_keeps_existing_speech_and_credential_handling(self):
        self.set_response("", "들리지 않은 발언")
        self.assertEqual(self.transcriber.transcribe(self.chunk), "")
        self.set_response("검토합니다", "fake-api-key")
        self.assertEqual(self.transcriber.transcribe(self.chunk), "검토합니다")
        silent = AudioChunk(0, np.zeros(SAMPLE_RATE, dtype=np.int16))
        self.client.models.generate_content.reset_mock()
        self.assertEqual(self.transcriber.transcribe(silent), "")
        self.client.models.generate_content.assert_not_called()
