"""Build a self-contained amd64 Ubuntu installer using uv and dpkg-deb."""

import platform
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.12.16"
PYTHON_MINOR = "3.12"
APP_VERSION = "1.3.0"
BACKEND_FILES = (
    "app.py",
    "audio_processor.py",
    "llm_summarizer.py",
    "meeting_session.py",
    "meeting_storage.py",
    "meeting_history.py",
)


def main() -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("Build on x86-64 Linux.")
    for tool in ("uv", "dpkg-deb"):
        if not shutil.which(tool):
            raise SystemExit(f"Install {tool} before building.")

    subprocess.run(["uv", "python", "install", PYTHON_VERSION], check=True)
    interpreter = Path(
        subprocess.check_output(
            ["uv", "python", "find", "--managed-python", PYTHON_VERSION], text=True
        ).strip()
    ).resolve()

    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="meet-docs-deb-") as temporary:
        package = Path(temporary) / "package"
        app = package / "opt/meet-docs"
        backend = app / "backend"
        backend.mkdir(parents=True)
        for file in BACKEND_FILES:
            shutil.copyfile(ROOT / file, backend / file)
        shutil.copyfile(ROOT / "ubuntu/desktop.py", app / "desktop.py")
        shutil.copyfile(ROOT / ".env.example", app / ".env.example")

        runtime = app / "runtime"
        shutil.copytree(
            interpreter.parent.parent,
            runtime,
            symlinks=True,
            ignore=shutil.ignore_patterns(
                "__pycache__", "site-packages", "pip", "pip3", "pip3.12"
            ),
        )
        dependencies = runtime / f"lib/python{PYTHON_MINOR}/site-packages"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(interpreter),
                "--target",
                str(dependencies),
                "--only-binary",
                ":all:",
                "--require-hashes",
                "-r",
                str(ROOT / "ubuntu/requirements.lock"),
            ],
            check=True,
        )

        launcher = package / "usr/bin/meet-docs"
        launcher.parent.mkdir(parents=True)
        launcher.write_text(
            '#!/bin/sh\nexec /usr/bin/python3 /opt/meet-docs/desktop.py "$@"\n',
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        desktop = package / "usr/share/applications/io.github.meetdocs.MeetDocs.desktop"
        desktop.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "ubuntu" / desktop.name, desktop)
        icon = ROOT / "ubuntu/icon.svg"
        if not icon.is_file():
            raise SystemExit("Ubuntu application icon is missing.")
        installed_icon = (
            package
            / "usr/share/icons/hicolor/scalable/apps/io.github.meetdocs.MeetDocs.svg"
        )
        installed_icon.parent.mkdir(parents=True)
        shutil.copyfile(icon, installed_icon)

        control = package / "DEBIAN/control"
        control.parent.mkdir()
        size = sum(item.stat().st_size for item in package.rglob("*") if item.is_file())
        control.write_text(
            f"""Package: meet-docs
Version: {APP_VERSION}
Section: sound
Priority: optional
Architecture: amd64
Maintainer: Meet Docs
Installed-Size: {(size + 1023) // 1024}
Depends: python3, python3-gi, gir1.2-gtk-3.0, gir1.2-webkit2-4.1, libportaudio2, libstdc++6
Description: Live Korean meeting transcription and minutes
 Record from the default microphone, display live Korean captions, and save
 meeting minutes and transcripts. Speech recognition and summarization use
 the Gemini Developer API with one API key.
""",
            encoding="utf-8",
        )

        artifact = Path(temporary) / f"meet-docs_{APP_VERSION}_amd64.deb"
        subprocess.run(
            [
                "dpkg-deb",
                "--root-owner-group",
                "-Zzstd",
                "-z10",
                "--threads-max=2",
                "--build",
                str(package),
                str(artifact),
            ],
            check=True,
        )
        shutil.move(artifact, dist / artifact.name)
        print(dist / artifact.name)


if __name__ == "__main__":
    main()
