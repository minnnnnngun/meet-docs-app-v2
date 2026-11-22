"""Ubuntu desktop window for the local meeting recorder."""

import os
import socket
import subprocess
import sys
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from secrets import token_urlsafe
from tempfile import TemporaryDirectory
from time import monotonic

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("WebKit2", "4.1")
from gi.repository import Gio, GLib, Gtk, WebKit2

APP_ID = "io.github.meetdocs.MeetDocs"
ROOT = Path(__file__).resolve().parent
DATA_DIR = (
    Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "meet-docs"
)


def local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class MeetDocs(Gtk.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APP_ID)
        self.server = None
        self.session_dir = None
        self.active_file = None
        self.started = False
        self.url = ""

    def do_activate(self) -> None:
        if self.get_active_window():
            self.get_active_window().present()
            return

        self.window = Gtk.ApplicationWindow(
            application=self, default_width=1180, default_height=760
        )
        self.window.set_title("실시간 회의록")
        self.window.set_icon_name(APP_ID)
        self.window.connect("delete-event", self.on_close)

        titlebar = Gtk.HeaderBar(title="실시간 회의록", show_close_button=True)
        open_output = Gtk.Button(label="저장 폴더 열기")
        open_output.connect("clicked", self.open_output)
        titlebar.pack_end(open_output)
        self.window.set_titlebar(titlebar)

        self.stack = Gtk.Stack()
        self.status = Gtk.Label(label="회의록 앱을 시작하는 중…", wrap=True)
        status_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=16,
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
        )
        status_box.pack_start(self.status, False, False, 0)
        self.restart = Gtk.Button(label="회의 화면 다시 시작")
        self.restart.set_no_show_all(True)
        self.restart.connect("clicked", self.launch_server)
        status_box.pack_start(self.restart, False, False, 0)
        self.stack.add_named(status_box, "status")
        context = WebKit2.WebContext.new_ephemeral()
        context.connect("download-started", self.on_download)
        self.webview = WebKit2.WebView.new_with_context(context)
        self.webview.connect("decide-policy", self.navigation_policy)
        self.stack.add_named(self.webview, "meeting")
        self.window.add(self.stack)
        self.window.show_all()
        self.window.present()
        self.launch_server()

    def launch_server(self, _button=None) -> None:
        self.stop_server()
        self.started = False
        self.restart.hide()
        self.status.set_text("회의록 앱을 시작하는 중…")
        self.stack.set_visible_child_name("status")
        try:
            self.start_server()
        except OSError:
            self.server_failed(
                "회의록 앱을 실행할 수 없습니다. 설치 상태를 확인하세요."
            )
            return
        GLib.timeout_add(250, self.check_server)

    def start_server(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        if DATA_DIR.is_symlink() or DATA_DIR.stat().st_uid != os.getuid():
            raise PermissionError("The application data directory must be user-owned.")
        DATA_DIR.chmod(0o700)
        self.session_dir = TemporaryDirectory(prefix="meet-docs-")
        self.active_file = Path(self.session_dir.name) / "recording"
        self.port = local_port()
        base_path = token_urlsafe(24)
        self.url = f"http://127.0.0.1:{self.port}/{base_path}/"
        self.health_path = f"/{base_path}/_stcore/health"
        self.deadline = monotonic() + 20
        self.next_health_check = 0.0
        self.failed_health_checks = 0

        env = os.environ.copy()
        env.update(
            MEET_DOCS_DATA_DIR=str(DATA_DIR),
            MEET_DOCS_ACTIVE_FILE=str(self.active_file),
        )
        self.server = subprocess.Popen(
            [
                str(ROOT / "runtime/bin/python3.12"),
                "-m",
                "streamlit",
                "run",
                str(ROOT / "backend/app.py"),
                "--server.address=127.0.0.1",
                f"--server.port={self.port}",
                f"--server.baseUrlPath={base_path}",
                "--server.headless=true",
                "--server.fileWatcherType=none",
                "--server.enableCORS=true",
                "--server.enableXsrfProtection=true",
                "--client.showErrorDetails=none",
                "--client.toolbarMode=minimal",
                "--browser.gatherUsageStats=false",
            ],
            cwd=DATA_DIR,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            umask=0o077,
        )

    def check_server(self) -> bool:
        if self.server.poll() is not None:
            self.server_failed(
                "회의 화면이 종료됐습니다. 다시 시작하면 저장된 기록을 복구할 수 있습니다."
            )
            return False
        if self.started and monotonic() < self.next_health_check:
            return True

        healthy = self.server_healthy()
        if self.started:
            self.next_health_check = monotonic() + 2
            self.failed_health_checks = 0 if healthy else self.failed_health_checks + 1
            if self.failed_health_checks < 3:
                return True
            self.server_failed(
                "회의 화면이 응답하지 않습니다. 다시 시작하면 저장된 기록을 복구할 수 있습니다."
            )
            return False
        if healthy:
            self.webview.load_uri(self.url)
            self.stack.set_visible_child_name("meeting")
            self.started = True
            self.next_health_check = monotonic() + 2
            return True
        if monotonic() >= self.deadline:
            self.server_failed("회의 화면이 응답하지 않습니다. 다시 시작하세요.")
            return False
        return True

    def server_healthy(self) -> bool:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=0.1)
        try:
            connection.request("GET", self.health_path)
            response = connection.getresponse()
            return response.status == 200 and response.read(16) == b"ok"
        except (OSError, HTTPException):
            return False
        finally:
            connection.close()

    def server_failed(self, message: str) -> None:
        self.status.set_text(message)
        self.stack.set_visible_child_name("status")
        self.restart.show()

    def navigation_policy(self, _view, decision, kind) -> bool:
        if kind == WebKit2.PolicyDecisionType.NAVIGATION_ACTION:
            action = decision.get_navigation_action()
            uri = action.get_request().get_uri()
            reloading = action.get_navigation_type() == WebKit2.NavigationType.RELOAD
            recording = self.active_file and self.active_file.exists()
            if self.url and uri.startswith(self.url) and not (recording and reloading):
                return False
        elif kind == WebKit2.PolicyDecisionType.RESPONSE:
            if not decision.is_main_frame_main_resource():
                return False
            if self.url and decision.get_request().get_uri().startswith(self.url):
                return False
        decision.ignore()
        return True

    def on_close(self, _window, _event) -> bool:
        running = self.server is not None and self.server.poll() is None
        if running and self.active_file and self.active_file.exists():
            self.message(
                "회의를 먼저 종료하세요.",
                "회의 종료를 누른 후 저장이 끝나면 창을 닫을 수 있습니다.",
            )
            return True
        return False

    def open_output(self, _button) -> None:
        output_dir = DATA_DIR / "output"
        try:
            output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            Gio.AppInfo.launch_default_for_uri(output_dir.as_uri(), None)
        except (OSError, GLib.Error):
            self.message("저장 폴더를 열 수 없습니다.", str(output_dir))

    def on_download(self, _context, download) -> None:
        if not self.url or not download.get_request().get_uri().startswith(self.url):
            download.cancel()
            return
        download.connect("decide-destination", self.download_destination)
        download.connect("failed", self.download_failed)

    def download_destination(self, download, suggested_name) -> bool:
        special = GLib.get_user_special_dir(GLib.UserDirectory.DIRECTORY_DOWNLOAD)
        folder = Path(special) if special else Path.home() / "Downloads"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            basename = Path(suggested_name).name
            destination = folder / basename
            if destination.suffix.lower() not in (".md", ".txt"):
                download.cancel()
                return True
            index = 2
            stem, suffix = destination.stem, destination.suffix
            while destination.exists() or destination.is_symlink():
                destination = folder / f"{stem}_{index}{suffix}"
                index += 1
            download.set_allow_overwrite(False)
            download.set_destination(destination.as_uri())
        except OSError:
            download.cancel()
            self.message("다운로드 폴더에 저장할 수 없습니다.", str(folder))
        return True

    def download_failed(self, _download, error) -> None:
        if not error.matches(
            WebKit2.download_error_quark(), WebKit2.DownloadError.CANCELLED_BY_USER
        ):
            self.message(
                "결과 파일을 다운로드하지 못했습니다.",
                "다운로드 폴더와 남은 공간을 확인하세요.",
            )

    def message(self, text, details) -> None:
        dialog = Gtk.MessageDialog(
            transient_for=self.window,
            modal=True,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text=text,
        )
        dialog.format_secondary_text(details)
        dialog.run()
        dialog.destroy()

    def stop_server(self) -> None:
        if self.server is not None and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()
        if self.session_dir is not None:
            self.session_dir.cleanup()
        self.server = None
        self.session_dir = None
        self.active_file = None


if __name__ == "__main__":
    os.umask(0o077)
    app = MeetDocs()
    try:
        result = app.run(sys.argv)
    finally:
        app.stop_server()
    sys.exit(result)
