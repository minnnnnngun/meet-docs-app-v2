import unittest
from dataclasses import replace

import numpy as np

from audio_processor import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    CHUNK_SECONDS,
    OVERLAP_SAMPLES,
    SAMPLE_RATE,
    _ButterworthSection,
    _ChunkBuffer,
    recognition_windows,
)


class AudioChunksTest(unittest.TestCase):
    def test_recognition_waits_only_at_active_boundary_and_keeps_one_overlap(self):
        source = np.random.default_rng(12).integers(
            -2000, 2000, size=SAMPLE_RATE * 6 + SAMPLE_RATE // 4, dtype=np.int16
        )
        source[2 * SAMPLE_RATE - SAMPLE_RATE // 4 : 2 * SAMPLE_RATE] = 0
        chunks = []
        buffer = _ChunkBuffer(SAMPLE_RATE, chunks.append)
        buffer.append(source[:, None])
        buffer.flush()

        first, joined, final = recognition_windows(chunks)
        self.assertIs(first, chunks[0])
        self.assertIs(final, chunks[3])
        self.assertEqual((joined.start_seconds, joined.end_seconds), (2, 6))
        self.assertEqual(joined.overlap_samples, OVERLAP_SAMPLES)
        np.testing.assert_array_equal(
            joined.samples,
            np.concatenate((chunks[1].samples, chunks[2].samples[OVERLAP_SAMPLES:])),
        )
        np.testing.assert_array_equal(
            joined.activity_samples, source[2 * SAMPLE_RATE : 6 * SAMPLE_RATE]
        )

    def test_recognition_flushes_held_audio_on_stop_and_does_not_join_a_gap(self):
        chunks = []
        buffer = _ChunkBuffer(SAMPLE_RATE, chunks.append, bandpass_enabled=False)
        buffer.append(np.full((2 * SAMPLE_RATE + 640, 1), 900, dtype=np.int16))
        buffer.flush()

        first, final = chunks
        self.assertIs(next(recognition_windows([first])), first)
        (joined,) = recognition_windows(chunks)
        self.assertEqual(joined.samples.size, 2 * SAMPLE_RATE + 640)
        self.assertIsNone(joined.activity_samples)
        after_gap = replace(final, start_seconds=4)
        held, skipped = recognition_windows([first, after_gap])
        self.assertIs(held, first)
        self.assertIs(skipped, after_gap)

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

    def test_bandpass_attenuates_out_of_band_tones_and_can_be_disabled(self):
        sample_time = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
        amplitudes = []
        for frequency in (100, 1000, 5000):
            source = np.rint(1000 * np.sin(2 * np.pi * frequency * sample_time)).astype(
                np.int16
            )
            enabled, disabled = [], []
            _ChunkBuffer(SAMPLE_RATE, enabled.append).append(source[:, None])
            _ChunkBuffer(SAMPLE_RATE, disabled.append, bandpass_enabled=False).append(
                source[:, None]
            )
            np.testing.assert_array_equal(disabled[0].samples, source)
            self.assertIsNone(disabled[0].activity_samples)
            np.testing.assert_array_equal(enabled[0].activity_samples, source)
            amplitudes.append(np.std(enabled[0].samples[SAMPLE_RATE:]))
        self.assertGreater(amplitudes[1], 650)
        self.assertLess(amplitudes[0], amplitudes[1] / 5)
        self.assertLess(amplitudes[2], amplitudes[1] / 2)

    def test_chunks_filter_each_sample_once_with_continuous_state(self):
        source = np.random.default_rng(7).integers(
            -2000, 2000, size=SAMPLE_RATE * 4 + 32, dtype=np.int16
        )
        chunks = []
        buffer = _ChunkBuffer(SAMPLE_RATE, chunks.append)
        buffer.append(source[:, None])
        buffer.flush()
        filtered = np.concatenate(
            [chunk.samples[chunk.overlap_samples :] for chunk in chunks]
        )
        highpass = _ButterworthSection(BANDPASS_LOW_HZ, highpass=True)
        lowpass = _ButterworthSection(BANDPASS_HIGH_HZ, highpass=False)
        whole = lowpass.apply(highpass.apply(source.astype(np.float64)))
        np.testing.assert_array_equal(filtered, np.rint(whole).astype(np.int16))
