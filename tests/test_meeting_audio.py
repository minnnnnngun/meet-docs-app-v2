import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from audio_processor import SAMPLE_RATE, AudioChunk
from meeting_audio import (
    TAPE_DIRECTORY,
    AudioTape,
    audio_available,
    delete_audio,
    read_clip,
)


class MeetingAudioTest(unittest.TestCase):
    def test_uses_fresh_unfiltered_samples_and_can_read_an_open_tape(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            tape = AudioTape(output)
            self.addCleanup(tape.close)
            self.assertFalse(audio_available(output, tape.audio_id))
            tape.append(
                AudioChunk(
                    0,
                    np.array([1, 2], dtype=np.int16),
                    activity_samples=np.array([1000, 2000], dtype=np.int16),
                )
            )
            tape.append(
                AudioChunk(2 / SAMPLE_RATE, np.array([2, 3000, 4000], np.int16), 1)
            )
            clip = read_clip(output, tape.audio_id, 1 / SAMPLE_RATE)
            self.assertEqual(clip.start_seconds, 0)
            self.assertEqual(clip.end_seconds, 4 / SAMPLE_RATE)
            np.testing.assert_array_equal(clip.samples, [1000, 2000, 3000, 4000])
            self.assertTrue(audio_available(output, tape.audio_id))

            tape.close()
            if os.name == "posix":
                storage = output / TAPE_DIRECTORY
                self.assertEqual(storage.stat().st_mode & 0o777, 0o700)
                self.assertEqual(
                    (storage / f"{tape.audio_id}.pcm").stat().st_mode & 0o777, 0o600
                )
            delete_audio(output, tape.audio_id)
            self.assertFalse(audio_available(output, tape.audio_id))

    def test_clips_exactly_ten_seconds_each_side_and_clamps_recording_edges(self):
        source = (np.arange(30 * SAMPLE_RATE, dtype=np.int32) % 30_000).astype(np.int16)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            tape = AudioTape(output)
            self.addCleanup(tape.close)
            tape.append(AudioChunk(0, source))
            tape.close()

            for elapsed, start, end in (
                (0.5, 0, 10.5),
                (15.5, 5.5, 25.5),
                (29.5, 19.5, 30),
            ):
                with self.subTest(elapsed=elapsed):
                    result = read_clip(output, tape.audio_id, elapsed)
                    self.assertEqual(result.start_seconds, start)
                    self.assertEqual(result.end_seconds, end)
                    np.testing.assert_array_equal(
                        result.samples,
                        source[round(start * SAMPLE_RATE) : round(end * SAMPLE_RATE)],
                    )
            for elapsed, radius in ((30, 10), (-1, 10), (1, 11), (1, float("nan"))):
                with (
                    self.subTest(elapsed=elapsed, radius=radius),
                    self.assertRaises(ValueError),
                ):
                    read_clip(output, tape.audio_id, elapsed, radius=radius)

    def test_preserves_gaps_and_does_not_write_duplicate_frames(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            tape = AudioTape(output)
            self.addCleanup(tape.close)
            tape.append(AudioChunk(2 / SAMPLE_RATE, np.array([1, 2], dtype=np.int16)))
            tape.append(
                AudioChunk(3 / SAMPLE_RATE, np.array([200, 3000, 4000], dtype=np.int16))
            )
            result = read_clip(output, tape.audio_id, 4 / SAMPLE_RATE)
            np.testing.assert_array_equal(result.samples, [0, 0, 1, 2, 3000, 4000])

    @unittest.skipUnless(os.name == "posix", "POSIX link checks")
    def test_refuses_traversal_and_linked_storage_or_tape(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (output / TAPE_DIRECTORY).symlink_to(outside, target_is_directory=True)
            with self.assertRaises(OSError):
                AudioTape(output)
            (output / TAPE_DIRECTORY).unlink()
            tape = AudioTape(output)
            tape.close()
            target = outside / "protected.pcm"
            target.write_bytes(b"\x01\x00")
            path = output / TAPE_DIRECTORY / f"{tape.audio_id}.pcm"
            path.unlink()
            path.symlink_to(target)

            self.assertFalse(audio_available(output, tape.audio_id))
            self.assertFalse(audio_available(output, "../outside/protected"))
            with self.assertRaises(OSError):
                delete_audio(output, tape.audio_id)
            with self.assertRaises(ValueError):
                read_clip(output, "../outside/protected", 0)
            self.assertEqual(target.read_bytes(), b"\x01\x00")
