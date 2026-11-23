"""Render verified quotes and the two separate saved-record review actions."""

import re
from html import escape
from pathlib import Path

import streamlit as st

from meeting_review import (
    ReviewDecision,
    ReviewRecord,
    ReviewTurn,
    approve_decision,
    confirm_turn,
)
from meeting_storage import Export


def highlight_unclear(text: str) -> str:
    html = escape(text)
    for label in ("[불명확]", "[확인 필요]"):
        html = html.replace(label, f'<mark class="md-unclear">{label}</mark>')
    return html


def utterance_html(
    turn: ReviewTurn, *, anchor: bool = False, confirmation: str = ""
) -> str:
    identity = f' id="review-turn-{turn.line}"' if anchor else ""
    badge = '<mark class="md-unclear">확인 필요</mark> ' if turn.unclear else ""
    interpretation = "<small>의도 추정</small> " if turn.intent else ""
    confirmed = (
        f"<p><strong>사용자가 확인한 발언</strong> {escape(confirmation)}</p>"
        if confirmation
        else ""
    )
    corrected = (
        f"<p><strong>표기 보정</strong> {highlight_unclear(turn.corrected)}</p>"
        if turn.intent and turn.corrected and turn.corrected != turn.intent
        else ""
    )
    return (
        f'<details class="md-utterance"{identity}>'
        f"<summary><time>{escape(turn.time)}</time>"
        f"<small>{escape(turn.speaker)}</small><br>{badge}{interpretation}"
        f"{highlight_unclear(turn.intent or turn.corrected or turn.original)}</summary>"
        f"<p><strong>전사 원문</strong> {highlight_unclear(turn.original)}</p>{corrected}{confirmed}</details>"
    )


def decision_quote(
    decision: ReviewDecision, turns: dict[int, ReviewTurn], *, link: bool
) -> str:
    turn = turns.get(decision.source_line) if decision.source_line is not None else None
    if (
        turn is None
        or turn.unclear
        or not decision.quote
        or decision.quote not in turn.original
    ):
        return ""
    label = escape(f"근거 발언 · {turn.time} · {turn.speaker}")
    if link:
        label = f'<a href="#review-turn-{turn.line}">{label}</a>'
    return f'<div class="md-quote">{label}<q>{escape(decision.quote)}</q></div>'


def approved_quote(record: ReviewRecord, index: int, *, link: bool) -> str:
    line = record.approvals.get(index)
    if line is None or line not in record.confirmations:
        return ""
    turn = next((turn for turn in record.turns if turn.line == line), None)
    if turn is None:
        return ""
    label = escape(f"사용자 확정 · 확인한 발언 · {turn.time}")
    if link:
        label = f'<a href="#review-turn-{turn.line}">{label}</a>'
    return f'<div class="md-quote"><strong>{label}</strong><q>{escape(record.confirmations[line])}</q></div>'


def _markdown_text(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+.!|<>])", r"\\\1", " ".join(text.split()))


def reviewed_download(export: Export, record: ReviewRecord) -> Export:
    if export.path.suffix != ".md" or not record.decisions:
        return export
    turns = {turn.line: turn for turn in record.turns}
    indexed = {
        (row.status, row.text): (i, row) for i, row in enumerate(record.decisions)
    }
    by_text = {row.text: (i, row) for i, row in indexed.values()}
    lines = []
    in_decisions = False
    for line in export.content.splitlines():
        if line.startswith("## "):
            in_decisions = line == "## 주요 논의 및 결정 사항"
        item = (
            re.fullmatch(r"- \*\*(확정|검토|확인 필요)\*\* — (.+)", line)
            if in_decisions
            else None
        )
        entry = (
            indexed.get((item[1], item[2]), by_text.get(item[2]))
            if item is not None
            else None
        )
        if entry is None:
            lines.append(line)
            continue
        index, decision = entry
        reviewed_line = record.approvals.get(index)
        approved_turn = turns.get(reviewed_line) if reviewed_line is not None else None
        confirmation = (
            record.confirmations.get(reviewed_line, "")
            if reviewed_line is not None
            else ""
        )
        status = (
            "사용자 확정"
            if approved_turn is not None and confirmation
            else decision.status
        )
        lines.append(f"- **{status}** — {_markdown_text(decision.text)}")
        if approved_turn is not None and confirmation:
            lines.append(
                f"  - 사용자가 확인한 발언 [{_markdown_text(approved_turn.time)}] "
                f"{_markdown_text(confirmation)}"
            )
        source = (
            turns.get(decision.source_line)
            if decision.source_line is not None
            else None
        )
        if (
            source
            and not source.unclear
            and decision.quote
            and decision.quote in source.original
        ):
            lines.append(
                f"  - 근거 발언 [{_markdown_text(source.time)}] {_markdown_text(source.speaker)}"
                f" — {_markdown_text(decision.quote)}"
            )
    return Export(export.path, "\n".join(lines) + "\n")


def saved_review_panel(output_dir: Path, stem: str, record: ReviewRecord) -> None:
    if not record.turns:
        return
    st.subheader("발언과 결정 검토")
    turns = {turn.line: turn for turn in record.turns}

    def turn_label(line: int) -> str:
        row = turns[line]
        return f"{row.time} · {row.speaker} · {(row.intent or row.corrected or row.original)[:56]}"

    selected_line = st.selectbox(
        "확인할 발언", tuple(turns), format_func=turn_label, key=f"review_turn_{stem}"
    )
    if selected_line is not None:
        turn = turns[selected_line]
        st.html(
            utterance_html(
                turn, confirmation=record.confirmations.get(selected_line, "")
            )
        )
        interpretation = st.text_area(
            "확인할 문장",
            value=record.confirmations.get(
                selected_line, turn.intent or turn.corrected or turn.original
            ),
            max_chars=520,
            key=f"review_text_{stem}_{selected_line}",
        )
        if st.button("발언 확인 저장", key=f"confirm_{stem}"):
            try:
                confirm_turn(output_dir, stem, selected_line, interpretation)
            except ValueError as exc:
                st.error(str(exc))
            except OSError:
                st.error("발언 확인을 저장하지 못했습니다.")
            else:
                st.rerun()

    candidates = tuple(
        index
        for index, decision in enumerate(record.decisions)
        if decision.status != "확정" and index not in record.approvals
    )
    if candidates and record.confirmations:
        index = st.selectbox(
            "확정할 결정 후보",
            candidates,
            format_func=lambda idx: record.decisions[idx].text,
            key=f"review_candidate_{stem}",
        )
        basis = st.selectbox(
            "결정의 근거로 사용할 확인 발언",
            tuple(sorted(record.confirmations)),
            format_func=turn_label,
            key=f"review_basis_{stem}",
        )
        if (
            st.button("결정 확정", key=f"approve_{stem}")
            and index is not None
            and basis is not None
        ):
            try:
                approve_decision(output_dir, stem, index, basis)
            except ValueError as exc:
                st.error(str(exc))
            except OSError:
                st.error("결정 확정을 저장하지 못했습니다.")
            else:
                st.rerun()

    st.subheader("발언 근거 찾기")
    st.html(
        "".join(
            utterance_html(
                turn, anchor=True, confirmation=record.confirmations.get(turn.line, "")
            )
            for turn in record.turns
        )
    )
