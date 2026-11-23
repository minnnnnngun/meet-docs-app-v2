"""Build the portable x86-64 Windows release with the app and runtime included."""

import hashlib
import shutil
import subprocess
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
APP_VERSION = "1.7.0"
PYTHON_VERSION = "3.12.16"
PYTHON_RELEASE = "20261110"
WINDOWS_TARGET = "x86_64-pc-windows-msvc"
PYTHON_SHA256 = "355ee61b8207b91574491b1c6dc9b0ebb4408fc56bfafe498f3ef439314b52dc"
PYTHON_ARCHIVE_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{PYTHON_RELEASE}/cpython-{PYTHON_VERSION}%2B{PYTHON_RELEASE}-"
    f"{WINDOWS_TARGET}-install_only_stripped.tar.gz"
)
APP_FILES = (
    ".env.example",
    ".streamlit/config.toml",
    "README.md",
    "SECURITY.md",
    "app.py",
    "audio_processor.py",
    "correction_memory.py",
    "correction_view.py",
    "licenses/Pretendard-OFL.txt",
    "llm_summarizer.py",
    "meeting_audio.py",
    "meeting_corrections.py",
    "meeting_history.py",
    "meeting_review.py",
    "meeting_session.py",
    "meeting_storage.py",
    "requirements.txt",
    "review_view.py",
    "run.bat",
    "static/PretendardVariable.woff2",
    "static/meet-docs-icon.png",
    "team_glossary.py",
    "workspace.css",
)


def main() -> None:
    if not shutil.which("uv"):
        raise SystemExit("Install uv before building.")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    name = f"meet-docs_{APP_VERSION}_windows_x64"
    with TemporaryDirectory(dir=dist, prefix=".windows-build-") as temporary:
        build = Path(temporary)
        bundle = build / name
        for file in APP_FILES:
            target = bundle / file
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / file, target)

        archive = build / "python.tar.gz"
        request = Request(PYTHON_ARCHIVE_URL, headers={"User-Agent": "Meet Docs build"})
        with urlopen(request, timeout=45) as response, archive.open("wb") as file:
            shutil.copyfileobj(response, file)
        with archive.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != PYTHON_SHA256:
                raise SystemExit("Windows Python archive checksum did not match.")
        with tarfile.open(archive) as python_archive:
            python_archive.extractall(build, filter="data")
        runtime = bundle / "runtime"
        (build / "python").rename(runtime)
        if not (runtime / "python.exe").is_file():
            raise SystemExit("Windows Python executable is missing.")

        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python-platform",
                WINDOWS_TARGET,
                "--python-version",
                PYTHON_VERSION,
                "--target",
                str(runtime / "Lib/site-packages"),
                "--only-binary",
                ":all:",
                "--require-hashes",
                "--link-mode",
                "copy",
                "-r",
                str(ROOT / "windows/requirements.lock"),
            ],
            check=True,
        )
        artifact = build / f"{name}.zip"
        with ZipFile(
            artifact, "w", compression=ZIP_DEFLATED, compresslevel=6
        ) as package:
            for file in sorted(bundle.rglob("*")):
                if file.is_file():
                    package.write(file, file.relative_to(build))
        destination = dist / artifact.name
        artifact.replace(destination)
        print(destination)


if __name__ == "__main__":
    main()
