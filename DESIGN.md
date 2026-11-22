# Design

## Source of truth

Active, November 22, 2026. The conversation, README, existing Streamlit workspace, and Ubuntu desktop launcher define the scope. The project had no existing design brief or reference mockup.

## Brand

Quiet, readable and practical. Recording, saving and recovery status remain visible. The workspace uses its own appearance and layout.

## Product goals

Start or stop recording from the main screen, scan live captions and read a structured meeting document. Search saved meetings by title or date to reopen a specific occurrence, read its full transcript, or download its files. Switch between light and dark, and recover minutes and captions after unexpected shutdown.

## Personas and jobs

A meeting organizer running the app on a laptop configures credentials before a meeting, monitors captions during the discussion, and shares the exported document afterward.

## Information architecture

The workspace has live and history views. Live has meeting setup in the sidebar, recording controls across the top, captions to the left, and a wider minutes document to the right. Recovery and export appear when needed. History offers a search field, an occurrence selector, and separate tabs for minutes and the full transcript.

## Design principles

Keep the recording controls in a consistent place. Group summaries into agenda, discussions and decisions, and action items. Keep the newest captions visible. Use the full meeting transcript for earlier captions.

## Visual language

The workspace uses system typography, restrained borders, a paper-like document surface, blue controls and compact timestamp labels. Light and dark palettes share typography, spacing and semantics. Action items display completion status only when the supplied summary specifies it.

## Components

The sidebar contains appearance, meeting topic, terminology, API keys and model. The workspace contains a status row, caption timeline, sectioned document, alerts and download controls. Native Streamlit controls handle actions and input.

## Accessibility

Keep control labels descriptive, preserve keyboard focus, and pair status color with text. Use heading levels for document sections and label completion indicators for assistive technology.

## Responsive behavior

Use a wide document pane beside a narrow caption timeline on desktop. Stack panels on narrow screens and let captions and long document text wrap.

## Interaction states

Idle, starting, recording, finishing, finished and failed states use different labels. Recovery blocks a new meeting until pending text has been saved. Incomplete summaries and delayed transcription surface inline with transcript access. History distinguishes an empty archive, an empty search, and an occurrence with only one saved file.

## Content voice

Use short Korean labels and explain errors where the user can act on them. API keys and meeting content remain out of status messages.

## Implementation constraints

Streamlit runs in the Ubuntu WebKit view and standard browsers. Rendering escapes all meeting content. Themes use scoped CSS and keep the current browser view's appearance selection. No network font or script is required.

## Open questions

- [ ] Whether to add direct Notion export is a product decision for a later release.
