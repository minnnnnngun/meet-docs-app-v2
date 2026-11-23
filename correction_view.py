"""Streamlit controls for user corrections and isolated background Gemini calls."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from html import escape
from pathlib import Path
from typing import TypeVar

import streamlit as st

from correction_memory import suggestions
from meeting_audio import audio_available
from meeting_corrections import recheck_turn
from meeting_review import ReviewTurn

T = TypeVar("T")


@st.cache_resource
def _workers() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="meeting-correction")


def background_action(
    key: str,
    label: str,
    work: Callable[[], T],
    result: Callable[[T], None],
    *,
    disabled: bool = False,
) -> None:
    job_key, notice_key = f"{key}_job", f"{key}_notice"
    future: Future[T] | None = st.session_state.get(job_key)
    if future is not None and future.done():
        st.session_state.pop(job_key)
        try:
            value = future.result()
        except Exception:  # noqa: BLE001 - avoid displaying backend errors and request contents.
            st.session_state[notice_key] = (
                "처리하지 못했습니다. 연결, 저장 상태와 최신 교정문을 확인한 뒤 다시 시도하세요."
            )
        else:
            result(value)
        future = None
    if message := st.session_state.pop(notice_key, None):
        st.warning(message)
    if st.button(label, key=key, disabled=disabled or future is not None):
        st.session_state[job_key] = _workers().submit(work)
        future = st.session_state[job_key]
    if future is not None:
        st.caption("처리 중입니다.")


def correction_editor(
    prefix: str,
    turn: ReviewTurn,
    current: str,
    *,
    output_dir: Path,
    audio_id: str | None,
    api_key: str,
    glossary: tuple[str, ...],
    save: Callable[[str], None],
) -> None:
    key = f"{prefix}_{turn.line}"
    text_key = f"{key}_text"
    proposal_key = f"{key}_proposal"
    original = turn.corrected or turn.original
    if audio_id and audio_available(output_dir, audio_id):
        hint = st.text_input("재인식 힌트", max_chars=120, key=f"{key}_hint")
        background_action(
            f"{key}_recheck",
            "힌트 입력 후 다시 듣기",
            lambda: recheck_turn(output_dir, audio_id, turn, hint, api_key),
            lambda proposal: st.session_state.update({proposal_key: proposal}),
            disabled=not api_key or not hint.strip() or api_key in hint,
        )
        if proposal := st.session_state.get(proposal_key):
            st.html(
                '<div class="md-quote"><strong>재인식 제안</strong>'
                f"<p>{escape(proposal)}</p></div>"
            )
            st.button(
                "제안을 교정문에 넣기",
                key=f"{key}_apply",
                on_click=st.session_state.update,
                args=({text_key: proposal[:520]},),
            )
    else:
        st.caption("재인식은 이 회의에 보관된 음성이 있을 때 사용할 수 있습니다.")

    for index, (found, standard) in enumerate(
        suggestions(current or original, glossary)
    ):
        st.button(
            f"표기 제안 · {found} → {standard}",
            key=f"{key}_suggest_{index}",
            on_click=st.session_state.update,
            args=({text_key: (current or original).replace(found, standard, 1)[:520]},),
        )
    text = (
        st.text_area(
            "교정문",
            value=(current or original)[:520]
            if text_key not in st.session_state
            else None,
            max_chars=520,
            key=text_key,
        )
        or ""
    )
    if st.button("교정문 저장", key=f"{key}_save", disabled=not text):
        if api_key and api_key in text:
            st.error("교정문에 API 키를 넣을 수 없습니다.")
            return
        try:
            save(text)
        except (OSError, ValueError):
            st.error("교정문을 저장하지 못했습니다. 발언과 입력 내용을 확인하세요.")
        else:
            st.rerun()
