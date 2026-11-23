"""Streamlit workspace for meeting capture, document review and export."""

import json
import os
import re
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

from correction_memory import correction_hints
from correction_view import correction_editor
from llm_summarizer import DEFAULT_GEMINI_MODEL
from meeting_history import (
    MINUTES,
    TRANSCRIPT,
    VERBATIM,
    ArchivedMeeting,
    list_meetings,
    read_meeting,
)
from meeting_review import ReviewRecord, read_review
from meeting_session import Export, MeetingDesk
from review_view import (
    approved_quote,
    decision_quote,
    highlight_unclear,
    reviewed_download,
    reviewed_transcript,
    saved_review_panel,
    utterance_html,
)
from team_glossary import read_glossary, save_glossary

ROOT = Path(__file__).resolve().parent
APP_ICON = ROOT / "static" / "meet-docs-icon.png"
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
STYLES = (ROOT / "workspace.css").read_text(encoding="utf-8")
MARKDOWN_SECTIONS = re.compile(r"^##[ \t]+(.+?)\s*$", re.MULTILINE)
MARKDOWN_ITEM = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)")
CHECKBOX = re.compile(r"^\[([ xX])\]\s+")
BOLD = re.compile(r"\*\*(.+?)\*\*")


def archive_transcript(text: str) -> None:
    with st.container(height=420, border=True):
        st.html(f'<div class="md-archive-transcript">{highlight_unclear(text)}</div>')


def configured_key(*names: str) -> str:
    for name in names:
        value = (os.getenv(name) or ENV_KEYS.get(name) or "").strip()
        if value:
            return value
    return ""


def workspace_banner(title: str, description: str, *, compact: bool = False) -> str:
    mode = " compact" if compact else ""
    wave = "<i></i>" * 5
    return (
        f'<header class="md-hero{mode}"><div><h1>{escape(title)}</h1>'
        f"<p>{escape(description)}</p></div>"
        '<div class="md-visual" aria-hidden="true">'
        f'<div class="md-wave">{wave}</div></div></header>'
    )


def minutes_document(
    title: str,
    minutes: str,
    message: str,
    review: ReviewRecord | None = None,
    *,
    saved: bool = False,
) -> str:
    """Build a document from escaped text and the supported Markdown markers."""
    sections = MARKDOWN_SECTIONS.split(minutes)[1:]
    rendered = []
    turns = {turn.line: turn for turn in review.turns} if review else {}
    rows = (
        {
            (decision.status, decision.text): (index, decision)
            for index, decision in enumerate(review.decisions)
        }
        if review
        else {}
    )
    by_text = {decision.text: (index, decision) for index, decision in rows.values()}
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
            quote = ""
            if (
                review
                and heading == "주요 논의 및 결정 사항"
                and (
                    decision_item := re.fullmatch(
                        r"\*\*(확정|검토|확인 필요)\*\* — (.+)", item
                    )
                )
            ):
                status, body_text = decision_item.groups()
                match = rows.get((status, body_text), by_text.get(body_text))
                if match is not None:
                    decision_index, decision = match
                    quote = decision_quote(decision, turns, link=saved)
                    approval = approved_quote(review, decision_index, link=saved)
                    quote += approval
                    item = f"**{'사용자 확정' if approval else decision.status}** — {body_text}"
            value = BOLD.sub(r"<strong>\1</strong>", highlight_unclear(item))
            if item.startswith("**확인 필요**"):
                value = value.replace(
                    "<strong>확인 필요</strong>",
                    '<mark class="md-unclear">확인 필요</mark>',
                    1,
                )
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
                    value = BOLD.sub(r"<strong>\1</strong>", highlight_unclear(task))
                    value += f'<span class="md-task-meta">{metadata}</span>'
                value = f'<input type="checkbox" disabled {state}><span>{value}</span>'
                items.append(f'<li class="md-task">{value}</li>')
            else:
                items.append(f"<li>{value}{quote}</li>")
        list_class = ' class="md-checklist"' if is_checklist else ""
        level = "h4" if title else "h3"
        rendered.append(
            f"<section><{level}>{escape(heading)}</{level}>"
            f"<ul{list_class}>{''.join(items)}</ul></section>"
        )
    content = (
        "".join(rendered)
        if rendered
        else f'<div class="md-empty"><small>{escape(message)}</small></div>'
    )
    subject = f"<h3>{escape(title)}</h3>" if title else ""
    return (
        '<article class="md-article">'
        '<header><span class="md-panel-icon" aria-hidden="true"></span>'
        f"<h2>{'회의록' if saved else '자동 회의록'}</h2></header>"
        f'<div class="md-article-body">{subject}{content}</div></article>'
    )


def render_downloads(exports: tuple[Export, ...]) -> None:
    exports = tuple(
        export for export in exports if export.path.suffix in (".md", ".txt")
    )
    if not exports:
        return
    for column, export in zip(st.columns(len(exports)), exports, strict=True):
        markdown = export.path.suffix == ".md"
        label = "회의록 다운로드" if markdown else "전체 녹취 다운로드"
        if export.path.name.endswith(f"_{VERBATIM}"):
            label = "전사 원문 다운로드"
        column.download_button(
            label,
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


def new_meeting_button(key: str, *, disabled: bool = False) -> None:
    if st.button(
        "새 회의록",
        key=key,
        icon=":material/add:",
        disabled=disabled,
    ):
        st.session_state.prepare_new_meeting = True
        st.rerun()


def history_view() -> None:
    meeting = desk.current()
    snapshot = meeting.snapshot() if meeting else None
    busy = snapshot is not None and snapshot.status in {
        "starting",
        "recording",
        "finishing",
    }
    unsaved = (
        snapshot is not None and snapshot.status == "finished" and not snapshot.saved
    )
    pending_recovery = desk.has_recovery(OUTPUT_DIR)
    new_meeting_button(
        "new_meeting_history", disabled=busy or unsaved or pending_recovery
    )
    if busy:
        st.info("진행 중인 회의는 실시간 회의 화면에서 확인할 수 있습니다.")
    elif unsaved or pending_recovery:
        st.info("실시간 회의 화면에서 이전 회의를 저장하거나 복구하세요.")

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

    with st.container(key="history_filters"):
        search_column, meeting_column = st.columns([1, 1.2], gap="small")
    search = search_column.text_input(
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

    selected_stem = meeting_column.selectbox(
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

    try:
        review = read_review(OUTPUT_DIR, selected_stem)
    except (OSError, ValueError):
        st.warning("이 회차의 검토 데이터를 읽을 수 없습니다.")
        review = None
    if review is not None:
        exports = tuple(
            reviewed_transcript(export, review)
            if export.path.name.endswith(f"_{TRANSCRIPT}")
            else export
            for export in exports
        )

    files = {
        kind: export
        for export in exports
        for kind in (MINUTES, TRANSCRIPT, VERBATIM)
        if export.path.name.endswith(f"_{kind}")
    }
    tab_labels = ["회의록", "전체 녹취"]
    if VERBATIM in files:
        tab_labels.append("전사 원문")
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        if minutes := files.get(MINUTES):
            document = (
                review.revised_minutes
                if review and review.revised_minutes is not None
                else minutes.content
            )
            if MARKDOWN_SECTIONS.search(document):
                st.html(
                    minutes_document(
                        selected.topic or "회의록",
                        document,
                        "",
                        review,
                        saved=True,
                    )
                )
            else:
                archive_transcript(document)
        else:
            st.info("이 회차의 회의록 파일이 없습니다.")
    with tabs[1]:
        if transcript := files.get(TRANSCRIPT):
            archive_transcript(transcript.content)
        else:
            st.info("이 회차의 전체 녹취 파일이 없습니다.")
    if verbatim := files.get(VERBATIM):
        with tabs[2]:
            st.caption("화자 번호는 각 음성 인식 구간 안에서만 유효합니다.")
            archive_transcript(verbatim.content)
    if review is not None:
        exports = tuple(reviewed_download(export, review) for export in exports)
    render_downloads(exports)
    if review is not None:
        saved_review_panel(
            OUTPUT_DIR,
            selected_stem,
            review,
            topic=selected.topic,
            api_key=gemini_key,
            model=model,
            glossary=tuple(glossary),
        )


st.set_page_config(page_title="실시간 회의록", page_icon=APP_ICON, layout="wide")


@st.cache_resource
def get_desk() -> MeetingDesk:
    return MeetingDesk()


desk = get_desk()
st.session_state.setdefault("navigation", "실시간 회의")
if st.session_state.pop("prepare_new_meeting", False):
    st.session_state.navigation = "실시간 회의"
    if desk.prepare_new_meeting(OUTPUT_DIR):
        st.session_state.meeting_topic = ""
        st.session_state.meeting_terms = ""
        st.session_state.retain_audio = False
        st.session_state.pop("start_error", None)
        st.session_state.pop("recovered_exports", None)
    else:
        st.session_state.new_meeting_notice = True

with st.sidebar:
    st.html(
        '<div class="md-brand">'
        f'<img class="md-brand-mark" src="app/static/{APP_ICON.name}" alt="">'
        "Meet Docs</div>"
    )
    view = st.segmented_control(
        "작업 화면",
        ("실시간 회의", "회의 기록"),
        format_func=lambda value: (
            ":material/mic: 실시간 회의"
            if value == "실시간 회의"
            else ":material/folder_open: 회의 기록"
        ),
        key="navigation",
        label_visibility="collapsed",
        width="stretch",
    )
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

    st.html('<h2 class="md-sidebar-title">회의 설정</h2>')
    topic = st.text_input(
        "회의 주제",
        max_chars=200,
        placeholder="예) 11월 제품 출시 점검",
        key="meeting_topic",
    )
    terms = st.text_area(
        "사내 전문 용어",
        max_chars=600,
        placeholder="예) 프로젝트명, 참석자 이름, 약어",
        key="meeting_terms",
    )
    bandpass_enabled = st.toggle(
        "회의 음성 필터",
        value=True,
        help="300–3,400Hz 대역을 중심으로 음성을 처리합니다.",
    )
    retain_audio = st.toggle(
        "재인식용 음성 보관",
        key="retain_audio",
        help="이 회의의 마이크 음성을 기기에 보관합니다. 구간 재인식을 요청하면 선택한 시각 주변만 Gemini로 전송합니다.",
    )
    try:
        glossary = read_glossary(DATA_DIR)
    except (OSError, ValueError):
        st.warning("팀 단어장을 읽을 수 없습니다.")
        glossary = {}
    with st.expander("팀 단어장"):
        st.caption(f"등록 {len(glossary)}개 · Gemini에는 단어 표기만 전송합니다.")
        with st.form("team_glossary"):
            glossary_content = st.text_area(
                "단어와 기기 내부 설명",
                value=json.dumps(glossary, ensure_ascii=False, indent=2),
                placeholder='{"프로젝트 알파": "내부 설명"}',
                height=160,
            )
            save_words = st.form_submit_button("팀 단어장 저장")
        if save_words:
            try:
                save_glossary(DATA_DIR, glossary_content)
            except ValueError:
                st.error("단어를 왼쪽, 설명을 오른쪽에 둔 JSON 형식을 확인하세요.")
            except OSError:
                st.error("팀 단어장을 저장하지 못했습니다.")
            else:
                st.rerun()
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
    st.caption("직접 저장한 짧은 교정 쌍은 다음 회의의 인식 힌트로 사용합니다.")

st.html(f'<span id="md-theme-{theme}" hidden></span><style>{STYLES}</style>')


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

    with st.container(key="meeting_controls"):
        start_column, stop_column, status_column = st.columns(
            [1.05, 1.05, 2], gap="small", vertical_alignment="center"
        )
    if start_column.button(
        "회의 시작",
        type="primary",
        icon=":material/play_arrow:",
        disabled=bool(busy or unsaved or pending_recovery),
        width="stretch",
    ):
        if not gemini_key:
            st.session_state.start_error = (
                "Gemini API 키를 입력하거나 .env 파일에 설정하세요."
            )
        else:
            try:
                remembered = correction_hints(DATA_DIR)
            except OSError:
                remembered = ()
                st.toast("최근 교정 후보를 읽지 못했습니다.")
            try:
                desk.start(
                    gemini_key=gemini_key,
                    topic=topic.strip(),
                    terms=terms.strip(),
                    model=model,
                    output_dir=OUTPUT_DIR,
                    bandpass_enabled=bandpass_enabled,
                    team_words=tuple(glossary),
                    retain_audio=retain_audio,
                    remembered_terms=remembered,
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
            icon=":material/stop_circle:",
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
        left, right = st.columns([1, 1.6], gap="medium")
        with left:
            captions = snapshot.captions if snapshot else ()
            shown = (
                f"<small>최근 {len(captions)}개</small>"
                if snapshot and snapshot.caption_count > len(captions)
                else ""
            )
            turns = (
                tuple(
                    caption.review_turn(line)
                    for line, caption in enumerate(
                        captions, start=snapshot.caption_count - len(captions) + 1
                    )
                )
                if snapshot
                else ()
            )
            rows = (
                "".join(
                    utterance_html(
                        turn, correction=snapshot.corrections.get(turn.line, "")
                    )
                    for turn in reversed(turns)
                )
                if snapshot
                else ""
            )
            body = rows or (
                '<div class="md-empty"><strong>아직 표시할 자막이 없습니다.</strong>'
                "<small>인식된 발언을 열면 전사 원문을 볼 수 있습니다.</small></div>"
            )
            st.html(
                '<section class="md-captions"><header><div class="md-panel-title">'
                '<span class="md-panel-icon caption" aria-hidden="true"></span>'
                f"<h2>실시간 자막</h2></div>{shown}</header>"
                f'<div class="md-captions-body">{body}</div></section>'
            )
            if meeting and snapshot and status == "recording" and turns:
                with st.expander("실시간 오인식 교정"):
                    by_line = {turn.line: turn for turn in reversed(turns)}
                    key = f"live_{meeting.started_at.isoformat()}"
                    line = st.selectbox(
                        "교정할 발언",
                        tuple(by_line),
                        format_func=lambda number: (
                            f"{by_line[number].time} · {by_line[number].original[:54]}"
                        ),
                        key=f"{key}_turn",
                    )
                    if line is not None:
                        correction_editor(
                            key,
                            by_line[line],
                            snapshot.corrections.get(line, ""),
                            output_dir=OUTPUT_DIR,
                            audio_id=snapshot.audio_id,
                            api_key=gemini_key,
                            glossary=tuple(glossary),
                            save=lambda text: meeting.save_live_correction(line, text),
                        )
                    if st.button("회의록 다시 정리"):
                        meeting.request_resummary()
                        st.toast("교정문으로 회의록을 다시 정리합니다.")
        with right:
            message = "새 자막이 쌓이면 30초마다 회의록을 갱신합니다."
            if snapshot and snapshot.exports and not snapshot.minutes:
                message = (
                    "회의록을 생성하지 못했습니다."
                    if snapshot.caption_count
                    else "인식된 발언이 없습니다."
                )
            title = meeting.topic if meeting else topic.strip()
            st.html(
                minutes_document(
                    title,
                    snapshot.minutes if snapshot else "",
                    message,
                    ReviewRecord(snapshot.evidence_turns, snapshot.review_decisions)
                    if snapshot
                    else None,
                )
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
            st.success("회의록, 전체 녹취 및 전사 원문을 저장했습니다.")
            new_meeting_button("new_meeting_saved", disabled=pending_recovery)
        elif st.button("파일 다시 저장") and meeting:
            meeting.retry_save()
            st.rerun()
        render_downloads(snapshot.exports)


if view == "회의 기록":
    st.html(
        workspace_banner(
            "회의 기록",
            "저장한 회차를 찾아 회의록과 발언 근거를 확인하세요.",
            compact=True,
        )
    )
    history_view()
else:
    st.html(
        workspace_banner(
            "실시간 회의록",
            "발언을 자막으로 보고 회의가 끝나면 회의록과 녹취를 저장하세요.",
        )
    )
    if st.session_state.pop("new_meeting_notice", False):
        st.warning("진행 중이거나 아직 저장되지 않은 회의를 먼저 마무리하세요.")
    meeting_view()
