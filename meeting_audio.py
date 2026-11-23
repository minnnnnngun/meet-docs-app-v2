"""Private, opt-in PCM tape for bounded re-transcription clips."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np

from audio_processor import SAMPLE_RATE, AudioChunk

TAPE_DIRECTORY = ".meeting-audio"
MAX_RADIUS_SECONDS = 10
SAMPLE_BYTES = 2
NO_FOLLOW = getattr(os, "O_NOFOLLOW", 0)
BINARY = getattr(os, "O_BINARY", 0)


def _filename(audio_id: str) -> str:
    if not isinstance(audio_id, str) or re.fullmatch("[0-9a-f]{32}", audio_id) is None:
        raise ValueError("Invalid audio recording ID")
    return f"{audio_id}.pcm"


def _require_directory(path: Path) -> None:
    info = path.lstat()
    reparse = getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    if not stat.S_ISDIR(info.st_mode) or reparse:
        raise OSError("The audio storage path must be a real directory")


@contextmanager
def _tape_directory(
    output_dir: Path, *, create: bool = False
) -> Iterator[tuple[Path, int | None]]:
    directory = output_dir / TAPE_DIRECTORY
    if create:
        output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_directory(output_dir)
    if os.name != "posix":
        if create:
            directory.mkdir(mode=0o700, exist_ok=True)
        _require_directory(directory)
        yield directory, None
        return

    flags = os.O_RDONLY | os.O_DIRECTORY | NO_FOLLOW
    parent = os.open(output_dir, flags)
    try:
        if create:
            try:
                os.mkdir(TAPE_DIRECTORY, mode=0o700, dir_fd=parent)
            except FileExistsError:
                pass
        descriptor = os.open(TAPE_DIRECTORY, flags, dir_fd=parent)
        try:
            info = os.fstat(descriptor)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise PermissionError("The audio storage directory must be private")
            yield directory, descriptor
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _open_tape(
    directory: Path, directory_fd: int | None, filename: str, *, create: bool = False
) -> int:
    flags = BINARY | NO_FOLLOW | getattr(os, "O_NONBLOCK", 0)
    flags |= os.O_RDWR | os.O_CREAT | os.O_EXCL if create else os.O_RDONLY
    path = directory / filename
    if directory_fd is None and not create:
        info = path.lstat()
        reparse = (
            getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        )
        if not stat.S_ISREG(info.st_mode) or reparse:
            raise OSError("The audio recording must be a regular file")
    descriptor = os.open(
        filename if directory_fd is not None else path,
        flags,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise OSError("The audio recording must be a regular file")
        if os.name == "posix" and (
            info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise PermissionError("The audio recording must be private")
        return descriptor
    except OSError:
        os.close(descriptor)
        raise


class AudioTape:
    """Append fresh, unfiltered samples with their position on the meeting timeline."""

    def __init__(self, output_dir: Path) -> None:
        self.audio_id = uuid4().hex
        with _tape_directory(output_dir, create=True) as (directory, directory_fd):
            descriptor = _open_tape(
                directory, directory_fd, _filename(self.audio_id), create=True
            )
        self._stream = os.fdopen(descriptor, "w+b", buffering=0)
        self._position = 0
        self._lock = Lock()

    def append(self, chunk: AudioChunk) -> None:
        if not isfinite(chunk.start_seconds) or chunk.start_seconds < 0:
            raise ValueError("Invalid audio chunk timestamp")
        if not 0 <= chunk.overlap_samples <= chunk.samples.size:
            raise ValueError("Invalid audio chunk overlap")
        fresh = chunk.activity_samples
        if fresh is None:
            fresh = chunk.samples[chunk.overlap_samples :]
        if fresh.ndim != 1 or fresh.size != chunk.samples.size - chunk.overlap_samples:
            raise ValueError("Invalid fresh audio samples")
        start = round(chunk.start_seconds * SAMPLE_RATE)
        with self._lock:
            skipped = max(0, self._position - start)
            if skipped >= fresh.size:
                return
            offset = start + skipped
            # Seeking past the last frame represents a gap as zero-valued PCM.
            self._stream.seek(offset * SAMPLE_BYTES)
            data = fresh[skipped:].astype("<i2", copy=False).tobytes()
            try:
                if self._stream.write(data) != len(data):
                    raise OSError("Could not write the complete audio chunk")
            except OSError:
                self._stream.truncate(self._position * SAMPLE_BYTES)
                raise
            self._position = offset + fresh.size - skipped

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                try:
                    os.fsync(self._stream.fileno())
                finally:
                    self._stream.close()


def read_clip(
    output_dir: Path, audio_id: str, elapsed_seconds: float, *, radius: float = 10
) -> AudioChunk:
    """Read up to ten seconds on either side of the requested meeting time."""
    filename = _filename(audio_id)
    if (
        not isfinite(elapsed_seconds)
        or elapsed_seconds < 0
        or not isfinite(radius)
        or not 0 < radius <= MAX_RADIUS_SECONDS
    ):
        raise ValueError("Invalid audio clip bounds")
    with _tape_directory(output_dir) as (directory, directory_fd):
        descriptor = _open_tape(directory, directory_fd, filename)
    with os.fdopen(descriptor, "rb") as stream:
        positions = os.fstat(stream.fileno()).st_size // SAMPLE_BYTES
        if elapsed_seconds >= positions / SAMPLE_RATE:
            raise ValueError("The requested time has no recorded audio")
        start = round(max(0, elapsed_seconds - radius) * SAMPLE_RATE)
        end = min(positions, round((elapsed_seconds + radius) * SAMPLE_RATE))
        stream.seek(start * SAMPLE_BYTES)
        payload = stream.read((end - start) * SAMPLE_BYTES)
    size = len(payload) // SAMPLE_BYTES
    if size == 0:
        raise ValueError("The requested clip is empty")
    samples = np.frombuffer(payload, dtype="<i2", count=size).astype(
        np.int16, copy=True
    )
    return AudioChunk(start / SAMPLE_RATE, samples)


def audio_available(output_dir: Path, audio_id: str) -> bool:
    try:
        filename = _filename(audio_id)
        with _tape_directory(output_dir) as (directory, directory_fd):
            descriptor = _open_tape(directory, directory_fd, filename)
            try:
                return os.fstat(descriptor).st_size >= SAMPLE_BYTES
            finally:
                os.close(descriptor)
    except (OSError, ValueError):
        return False


def delete_audio(output_dir: Path, audio_id: str) -> None:
    """Delete the specified local tape through its validated recording ID."""
    filename = _filename(audio_id)
    with _tape_directory(output_dir) as (directory, directory_fd):
        descriptor = _open_tape(directory, directory_fd, filename)
        os.close(descriptor)
        os.unlink(
            filename if directory_fd is not None else directory / filename,
            dir_fd=directory_fd,
        )
