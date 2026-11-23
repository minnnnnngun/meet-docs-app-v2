# Design

## Source of truth

Active, November 23, 2026. The supplied dashboard reference informs the palette, soft shapes, and translucent surfaces. The existing Streamlit workspace, Ubuntu launcher, and saved meeting workflow define the layout and controls.

## Brand

Quiet, readable and practical. Meet Docs uses a white waveform on a pastel glass-like shape as its app icon and sidebar mark. Recording, saving and recovery status remain easy to find.

## Product goals

Start or stop recording from the main screen, scan live captions and read a structured meeting document. Search saved meetings by title or date to reopen a specific occurrence, read its full transcript, or download its files. Switch between light and dark, and recover minutes and captions after unexpected shutdown.

## Personas and jobs

A meeting organizer running the app on a laptop configures credentials before a meeting, monitors captions during the discussion, and shares the exported document afterward.

## Information architecture

The sidebar groups the Meet Docs mark, live and history navigation, appearance, and meeting settings. The live view opens with a title, then a recording control card, captions on the left, and a wider minutes document on the right. Recovery and export appear when needed. History offers a search field, a meeting selector, tabs for minutes and transcript files, and saved evidence and approval controls.

## Design principles

Keep the recording controls in a consistent place. Group summaries into agenda, discussions and decisions, and action items. Keep the newest captions visible. Use the full meeting transcript for earlier captions.

## Visual language

The workspace uses the bundled Pretendard Variable font, rounded cards, a violet primary button, and compact timestamp labels. The light palette blends sky blue, lilac, and peach across the shell and hero. The dark palette uses ink, plum, and a muted blue gradient. Reading surfaces are more opaque than the shell to preserve contrast. A static waveform sits on a glass-like shape in the header. Action items display completion status when the summary specifies it.

## Components

The sidebar contains navigation, appearance, meeting topic, terminology, audio filter, glossary, and Gemini settings. The workspace contains the recording control card, caption timeline, sectioned document, alerts, and downloads. Native Streamlit controls handle actions and input. Workspace colors and component styles live in `workspace.css`.

## Accessibility

Target WCAG 2.2 AA text contrast. Keep control labels descriptive, preserve visible keyboard focus, and pair status color with text. Use heading levels for document sections and label completion indicators for assistive technology. The hero artwork is static and hidden from assistive technology.

## Responsive behavior

Use a wide document pane beside the caption timeline on desktop. Stack the panels below 900 px and let the recording controls wrap below 620 px. Compact the hero and let captions and long document text wrap.

## Interaction states

Idle, starting, recording, finishing, finished and failed states use different labels. Recovery blocks a new meeting until pending text has been saved. Incomplete summaries and delayed transcription surface inline with transcript access. History distinguishes an empty archive, an empty search, and an occurrence with only one saved file.

## Content voice

Use short Korean labels and explain errors where the user can act on them. API keys and meeting content remain out of status messages.

## Implementation constraints

Streamlit runs in Ubuntu WebKit and standard browsers. Rendering escapes all meeting content. Themes use local CSS and keep the appearance selection in the current view. Styling and artwork load from the app package. Verify both palettes, the narrow layout, and the saved-evidence route using synthetic data.

## Open questions

- [ ] Whether to add direct Notion export is a product decision for a later release.
