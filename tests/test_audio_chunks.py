import unittest

import numpy as np

from audio_processor import CHUNK_SECONDS, OVERLAP_SAMPLES, SAMPLE_RATE, _ChunkBuffer


class AudioChunksTest(unittest.TestCase):
    def test_two_second_chunks_overlap_and_flush_at_the_device_sample_rate(self):
        for input_rate in (SAMPLE_RATE, 48_000):
            with self.subTest(sample_rate=input_rate):
                chunks = []
                audio = _ChunkBuffer(input_rate, chunks.append)
                self.assertEqual(CHUNK_SECONDS, 2)
                source = np.full((input_rate * 2, 1), 800, dtype=np.int16)
                audio.append(source[: input_rate - 7])
                self.assertFalse(chunks)
                audio.append(source[input_rate - 7 :])
                audio.append(source)
                audio.append(source[: input_rate // 4])
                audio.flush()

                self.assertEqual([chunk.start_seconds for chunk in chunks], [0, 2, 4])
                self.assertEqual([chunk.end_seconds for chunk in chunks], [2, 4, 4.25])
                self.assertEqual(
                    [chunk.samples.size for chunk in chunks],
                    [32_000, 32_000 + OVERLAP_SAMPLES, 4_000 + OVERLAP_SAMPLES],
                )
                np.testing.assert_array_equal(
                    chunks[0].samples[-OVERLAP_SAMPLES:],
                    chunks[1].samples[:OVERLAP_SAMPLES],
                )
                np.testing.assert_array_equal(
                    chunks[1].samples[-OVERLAP_SAMPLES:],
                    chunks[2].samples[:OVERLAP_SAMPLES],
                )
