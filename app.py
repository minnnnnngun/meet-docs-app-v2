"""Streamlit workspace for meeting capture, document review and export."""

import os
import re
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

from llm_summarizer import DEFAULT_GEMINI_MODEL
from meeting_history import ArchivedMeeting, list_meetings, read_meeting
from meeting_session import Export, MeetingDesk, elapsed_time

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("MEET_DOCS_DATA_DIR") or ROOT)
OUTPUT_DIR = DATA_DIR / "output"
try:
    ENV_KEYS = dotenv_values(DATA_DIR / ".env", interpolate=False, encoding="utf-8-sig")
except (OSError, UnicodeError):
    ENV_KEYS = {}
MODELS = (DEFAULT_GEMINI_MODEL, "gemini-2.5-flash")
STATUS_LABELS = {
    "starting": "마이크 연결 중",
    "recording": "녹음 중",
    "finishing": "마지막 발언 처리 및 파일 저장 중",
    "finished": "회의 종료",
    "failed": "마이크를 시작하지 못했습니다",
}
THEMES = {
    "light": "밝게",
    "dark": "어둡게",
}
PALETTES = {
    "light": {
        "bg": "#f4f6f9",
        "paper": "#ffffff",
        "field": "#f8faff",
        "muted": "#536176",
        "fg": "#172139",
        "stroke": "#dbe2ed",
        "wash": "#edf2fa",
        "accent": "#294d9d",
        "monitor": "#9d2745",
    },
    "dark": {
        "bg": "#10141b",
        "paper": "#19212d",
        "field": "#101824",
        "muted": "#aebacd",
        "fg": "#edf3ff",
        "stroke": "#374457",
        "wash": "#243045",
        "accent": "#aac6ff",
        "monitor": "#ff9eb2",
    },
}
CSS = """
.stApp, [data-testid="stAppViewContainer"] {
  background: var(--md-bg); color: var(--md-fg); color-scheme: MD_THEME;
}
[data-testid="stHeader"] { background: var(--md-bg); }
[data-testid="stSidebar"] { background: var(--md-paper); border-right: 1px solid var(--md-stroke); }
.stMainBlockContainer, [data-testid="stMainBlockContainer"] { padding-top: 3rem; }
.stApp :is(h1,h2,h3,p,label,summary,textarea,input,button,[role="combobox"]) { color: var(--md-fg); }
.stApp [data-testid="stCaptionContainer"] p { color: var(--md-muted); }
.stApp button:not([kind="header"]):not([kind="primary"]),
.stApp [data-testid="stExpander"] details {
  background: var(--md-paper); color: var(--md-fg); border-color: var(--md-stroke);
}
.stApp :is([data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
  [data-testid="stSelectbox"] [role="group"], [data-baseweb="base-input"],
  [data-baseweb="textarea"], [data-baseweb="select"] > div) {
  background: var(--md-field); color: var(--md-fg); border-color: var(--md-stroke);
}
.stApp :is([data-testid="stTextInputRootElement"], [data-testid="stSelectbox"] [role="group"])
  button:not(:hover) { background: transparent; }
.stApp :is(input, textarea)::placeholder { color: var(--md-muted); opacity: 1; }
.stApp [data-testid="stExpander"] summary {
  background: var(--md-wash); color: var(--md-fg);
}
.stApp [data-testid="stExpander"] summary:hover { background: var(--md-field); }
.stApp [data-testid="stSidebar"] :is([data-testid="stIconMaterial"], [data-testid="stTooltipIcon"] svg),
.stApp [data-testid="stBaseButton-headerNoPadding"] svg { color: var(--md-muted); }
div[data-baseweb="popover"], div[data-baseweb="popover"] [role="listbox"],
[role="listbox"], [role="listbox"] [role="option"] {
  background: var(--md-paper); color: var(--md-fg); border-color: var(--md-stroke);
}
[role="listbox"] [role="option"]:is(:hover, [aria-selected="true"]) {
  background: var(--md-wash); color: var(--md-fg);
}
.stApp button[kind="primary"] {
  background: #3158ad; border-color: #3158ad; color: #fff;
}
.stApp button[kind="primary"] p { color: #fff; }
.stApp button:is([aria-pressed="true"], [role="radio"][aria-checked="true"]) {
  background: var(--md-wash); border-color: var(--md-accent);
}
.stApp button:is([aria-pressed="true"], [role="radio"][aria-checked="true"]) :is(p, span) {
  color: var(--md-accent);
}
.stApp :focus-visible { outline: 2px solid var(--md-accent); outline-offset: 2px; }
.stApp [data-testid="stAlert"] > div { background: var(--md-wash); color: var(--md-fg); }
.stApp [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--md-stroke); }
.stApp hr { border-color: var(--md-stroke); }
.md-heading { border-bottom: 1px solid var(--md-stroke); padding-bottom: 1.15rem; }
.md-heading h1 { padding: 0; font-size: clamp(1.65rem, 2.9vw, 2.35rem); letter-spacing: -.045em; }
.md-heading p { margin: .55rem 0 0; color: var(--md-muted); font-size: .9rem; }
.md-status { display: inline-flex; align-items: center; gap: .55rem; padding: .3rem .7rem;
  border-radius: 2rem; background: var(--md-wash); color: var(--md-fg); font-size: .8rem; }
.md-status::before { content: ''; width: .5rem; height: .5rem; border-radius: 50%;
  background: var(--md-muted); }
.md-status.recording::before { background: var(--md-monitor); }
.md-panel-title { font-size: 1rem; font-weight: 720; margin: 0; }
.md-article, .md-captions {
  background: var(--md-paper); color: var(--md-fg); border: 1px solid var(--md-stroke);
  border-radius: .85rem; min-height: 340px; overflow-wrap: anywhere;
}
.md-article { padding: clamp(1.15rem, 3vw, 2.75rem); }
.md-captions { padding: 1rem; max-height: 65vh; overflow-y: auto; }
.md-article header { border-bottom: 1px solid var(--md-stroke); margin-bottom: 1.35rem;
  padding-bottom: 1.1rem; }
.md-article h3 { font-size: clamp(1.2rem, 2.4vw, 1.8rem); padding: 0; letter-spacing: -.045em; }
.md-article h4 { color: var(--md-fg); font-size: 1rem; padding: 0; font-weight: 700; }
.md-article section { padding: .85rem 0; }
.md-article section + section, .md-caption + .md-caption {
  border-top: 1px solid var(--md-stroke);
}
.md-article small, .md-captions small { color: var(--md-muted); font-size: .76rem; }
.md-article ul { margin: .55rem 0 0; }
.md-article li { line-height: 1.7; }
.md-article .md-checklist { list-style: none; padding: 0; }
.md-article .md-task { display: flex; align-items: flex-start; gap: .6rem; margin: .45rem 0; }
.md-task-meta { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .2rem; }
.md-task-meta small { background: var(--md-wash); border-radius: .35rem; padding: .1rem .4rem; }
.md-article .md-checklist input { width: 1rem; height: 1rem; margin: .3rem 0 0;
  flex: 0 0 1rem; accent-color: var(--md-accent); }
.md-caption { padding: .65rem 0; font-size: .86rem; line-height: 1.55; white-space: pre-wrap; }
.md-caption time { display: block; margin-bottom: .13rem; color: var(--md-muted);
  font-size: .73rem; font-variant-numeric: tabular-nums; }
.md-archive-transcript {
  white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.65;
}
@media (max-width: 720px) {
  .st-key-workspace [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  .st-key-workspace [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: 100%; }
  .md-captions { max-height: 20rem; min-height: 10rem; }
}
"""
MARKDOWN_SECTIONS = re.compile(r"^##[ \t]+(.+?)\s*$", re.MULTILINE)
MARKDOWN_ITEM = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)")
CHECKBOX = re.compile(r"^\[([ xX])\]\s+")
BOLD = re.compile(r"\*\*(.+?)\*\*")


def configured_key(*names: str) -> str:
    for name in names:
        value = (os.getenv(name) or ENV_KEYS.get(name) or "").strip()
        if value:
            return value
    return ""


def minutes_document(title: str, minutes: str, message: str) -> str:
    """Build a document from escaped text and the supported Markdown markers."""
    sections = MARKDOWN_SECTIONS.split(minutes)[1:]
    rendered = []
    for index, (heading, body) in enumerate(
        zip(sections[::2], sections[1::2], strict=True)
    ):
        items = []
        is_checklist = index == 2
        for line in body.splitlines():
            item = MARKDOWN_ITEM.sub("", line.strip())
            if not item:
                continue
            check = CHECKBOX.match(item)
            if check:
                item = item[check.end() :]
            value = BOLD.sub(r"<strong>\1</strong>", escape(item))
            if is_checklist and check:
                state = (
                    'checked aria-label="완료"'
                    if check[1] != " "
                    else 'aria-label="미완료"'
                )
                task, *details = item.split(" | ")
                if details and all(
                    field.startswith(("담당자:", "기한:")) for field in details
                ):
                    metadata = "".join(
                        f"<small>{escape(field)}</small>" for field in details
                    )
                    value = BOLD.sub(r"<strong>\1</strong>", escape(task))
                    value += f'<span class="md-task-meta">{metadata}</span>'
                value = f'<input type="checkbox" disabled {state}><span>{value}</span>'
                items.append(f'<li class="md-task">{value}</li>')
            else:
                items.append(f"<li>{value}</li>")
        list_class = ' class="md-checklist"' if is_checklist else ""
        rendered.append(
            f"<section><h4>{escape(heading)}</h4><ul{list_class}>{''.join(items)}</ul></section>"
        )
    content = "".join(rendered) if rendered else f"<small>{escape(message)}</small>"
    return f'<article class="md-article"><header><h3>{escape(title)}</h3></header>{content}</article>'


def render_downloads(exports: tuple[Export, ...]) -> None:
    if not exports:
        return
    for column, export in zip(st.columns(len(exports)), exports, strict=True):
        markdown = export.path.suffix == ".md"
        column.download_button(
            "회의록 다운로드" if markdown else "전체 녹취 다운로드",
            export.content,
            export.path.name,
            mime="text/markdown" if markdown else "text/plain",
            key=f"download_{export.path.name}",
            on_click="ignore",
            width="stretch",
        )


def history_label(meeting: ArchivedMeeting) -> str:
    return (
        f"{meeting.topic or '제목 없는 회의'} · {meeting.round_number}회차"
        f" · {meeting.started_at:%Y-%m-%d %H:%M}"
    )


def history_view() -> None:
    meeting = desk.current()
    if meeting and meeting.snapshot().status in {"starting", "recording", "finishing"}:
        st.info("진행 중인 회의는 실시간 회의 화면에서 확인할 수 있습니다.")

    try:
        meetings = list_meetings(OUTPUT_DIR)
    except OSError:
        st.error("회의 기록을 불러오지 못했습니다. 저장 폴더 권한을 확인하세요.")
        return
    if not meetings:
        st.info(
            "저장된 회의 기록이 없습니다. 회의를 종료하고 저장하면 여기에 표시됩니다."
        )
        return

    search = st.text_input(
        "제목 또는 날짜로 검색", placeholder="예) 주간 점검, 2026-11-22"
    )
    matches = {
        record.stem: record
        for record in meetings
        if all(
            word in f"{history_label(record)} {record.stem}".casefold()
            for word in search.casefold().split()
        )
    }
    if not matches:
        st.info("검색 조건에 맞는 회의가 없습니다.")
        return

    selected_stem = st.selectbox(
        "회의 회차",
        tuple(matches),
        format_func=lambda stem: history_label(matches[stem]),
    )
    if selected_stem is None:
        return

    selected = matches[selected_stem]
    try:
        exports = read_meeting(OUTPUT_DIR, selected)
    except (OSError, UnicodeError):
        st.error("선택한 회의를 열지 못했습니다. 파일과 저장 폴더를 확인하세요.")
        return
    if not exports:
        st.info("선택한 회의의 파일이 저장 폴더에 없습니다.")
        return

    files = {export.path.suffix: export for export in exports}
    minutes_tab, transcript_tab = st.tabs(("회의록", "전체 녹취"))
    with minutes_tab:
        if minutes := files.get(".md"):
            if MARKDOWN_SECTIONS.search(minutes.content):
                st.html(
                    minutes_document(selected.topic or "회의록", minutes.content, "")
                )
            else:
                st.text(minutes.content)
        else:
            st.info("이 회차의 회의록 파일이 없습니다.")
    with transcript_tab:
        if transcript := files.get(".txt"):
            with st.container(height=420, border=True):
                st.html(
                    '<div class="md-archive-transcript">'
                    f"{escape(transcript.content)}</div>"
                )
        else:
            st.info("이 회차의 전체 녹취 파일이 없습니다.")
    render_downloads(exports)


st.set_page_config(page_title="실시간 회의록", page_icon="🎙️", layout="wide")


@st.cache_resource
def get_desk() -> MeetingDesk:
    return MeetingDesk()


desk = get_desk()
with st.sidebar:
    st.header("회의 설정")
    initial_theme = st.query_params.get("theme", "light")
    theme = (
        st.segmented_control(
            "화면 모드",
            THEMES,
            default=initial_theme if initial_theme in THEMES else "light",
            format_func=THEMES.__getitem__,
            key="appearance",
            width="stretch",
        )
        or "light"
    )
    if theme != initial_theme:
        st.query_params["theme"] = theme

    topic = st.text_input(
        "회의 주제", max_chars=200, placeholder="예) 11월 제품 출시 점검"
    )
    terms = st.text_area(
        "사내 전문 용어", max_chars=600, placeholder="예) 프로젝트명, 참석자 이름, 약어"
    )
    env_gemini = configured_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    with st.expander("Gemini 연결", expanded=not env_gemini):
        gemini_key = (
            st.text_input(
                "Gemini API 키",
                type="password",
                placeholder="환경 설정에서 읽음" if env_gemini else None,
            ).strip()
            or env_gemini
        )
        model = st.selectbox(
            "회의록 모델",
            MODELS,
            help="한국어 음성 인식은 Gemini 3.5 Flash-Lite를 사용합니다.",
        )
    st.caption("음성과 회의록 요약 자료는 Gemini로 전송합니다.")

variables = "".join(f"--md-{name}:{color};" for name, color in PALETTES[theme].items())
st.html(f"<style>:root{{{variables}}}{CSS.replace('MD_THEME', theme)}</style>")
view = st.segmented_control(
    "작업 화면",
    ("실시간 회의", "회의 기록"),
    default="실시간 회의",
    label_visibility="collapsed",
)


@st.fragment(run_every=1)
def meeting_view() -> None:
    meeting = desk.current()
    snapshot = meeting.snapshot() if meeting else None
    status = snapshot.status if snapshot else "idle"
    busy = status in ("starting", "recording", "finishing")
    unsaved = (
        status == "finished" and snapshot and snapshot.exports and not snapshot.saved
    )
    pending_recovery = desk.has_recovery(OUTPUT_DIR)

    start_column, stop_column, status_column = st.columns(
        [1.25, 1.25, 4], vertical_alignment="center"
    )
    if start_column.button(
        "회의 시작",
        type="primary",
        disabled=bool(busy or unsaved or pending_recovery),
        width="stretch",
    ):
        if not gemini_key:
            st.session_state.start_error = (
                "Gemini API 키를 입력하거나 .env 파일에 설정하세요."
            )
        else:
            try:
                desk.start(
                    gemini_key=gemini_key,
                    topic=topic.strip(),
                    terms=terms.strip(),
                    model=model,
                    output_dir=OUTPUT_DIR,
                )
            except (OSError, RuntimeError):
                st.session_state.start_error = "회의를 시작하지 못했습니다. 저장 폴더 권한과 복구 상태를 확인하세요."
            else:
                st.session_state.pop("start_error", None)
                st.session_state.pop("recovered_exports", None)
                st.rerun()

    if (
        stop_column.button(
            "회의 종료",
            disabled=status not in ("starting", "recording"),
            width="stretch",
        )
        and meeting
    ):
        meeting.request_stop()
        st.rerun()

    label = STATUS_LABELS.get(status, "회의 시작 대기 중")
    status_column.html(
        f'<span class="md-status {escape(status)}" role="status">{escape(label)}</span>'
    )
    if st.session_state.get("start_error"):
        st.error(st.session_state.start_error)

    if pending_recovery:
        st.warning(
            "저장하지 못한 회의가 있습니다. 복구한 뒤 새 회의를 시작할 수 있습니다."
        )
        if st.button("이전 회의 복구"):
            try:
                st.session_state.recovered_exports = desk.recover(OUTPUT_DIR)
            except OSError:
                st.error(
                    "이전 회의를 저장하지 못했습니다. 저장 폴더 권한과 남은 공간을 확인하세요."
                )
            else:
                st.rerun()

    recovered_exports = st.session_state.get("recovered_exports")
    if recovered_exports:
        st.success("이전 회의에서 복구한 파일을 저장 폴더에 저장했습니다.")
        render_downloads(recovered_exports)

    with st.container(key="workspace"):
        left, right = st.columns([1, 2.2], gap="medium")
        with left:
            st.html('<h2 class="md-panel-title">실시간 자막</h2>')
            if snapshot and snapshot.captions:
                count = len(snapshot.captions)
                preview = (
                    f"<small>최근 {count}개 자막</small>"
                    if snapshot.caption_count > count
                    else ""
                )
                rows = "".join(
                    f'<div class="md-caption"><time>{elapsed_time(caption.seconds)}</time>{escape(caption.text)}</div>'
                    for caption in reversed(snapshot.captions)
                )
            else:
                preview = (
                    "<small>말뜻을 유지하며 표기를 보정한 자막이 표시됩니다.</small>"
                )
                rows = ""
            st.html(
                f'<section class="md-captions" aria-label="자막 타임라인">{preview}{rows}</section>'
            )
        with right:
            st.html('<h2 class="md-panel-title">자동 회의록</h2>')
            message = "새 자막이 쌓이면 30초마다 회의록을 갱신합니다."
            if snapshot and snapshot.exports and not snapshot.minutes:
                message = (
                    "회의록을 생성하지 못했습니다."
                    if snapshot.caption_count
                    else "인식된 발언이 없습니다."
                )
            title = (meeting.topic if meeting else topic.strip()) or "회의록"
            st.html(
                minutes_document(title, snapshot.minutes if snapshot else "", message)
            )

    if not snapshot:
        return
    notices = []
    if busy and snapshot.queue_size:
        notices.append(f"인식 대기 구간 {snapshot.queue_size}개")
    if snapshot.lost_chunks:
        notices.append(f"인식 누락 구간 {snapshot.lost_chunks}개")
    if status == "finished" and snapshot.summary_pending:
        notices.append(f"회의록에 반영되지 않은 자막 {snapshot.summary_pending}개")
    if notices:
        st.caption(" · ".join(notices))
    for warning in snapshot.warnings:
        st.warning(warning)

    if status == "finished" and snapshot.exports:
        if snapshot.saved:
            st.success("저장 폴더에 회의록과 전체 녹취를 저장했습니다.")
        elif st.button("파일 다시 저장") and meeting:
            meeting.retry_save()
            st.rerun()
        render_downloads(snapshot.exports)


if view == "회의 기록":
    st.html('<div class="md-heading"><h1>회의 기록</h1></div>')
    history_view()
else:
    st.html(
        '<div class="md-heading"><h1>실시간 회의록</h1>'
        "<p>자막을 확인하고 회의록과 전체 녹취를 저장하세요.</p></div>"
    )
    meeting_view()
