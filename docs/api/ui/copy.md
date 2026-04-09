# ui.copy

Centralized user-facing copy strings for all UI surfaces.

All text shown to the user in notifications, live status, and gap-fill
prompts is defined in `taskclf.ui.copy` so that labeling conventions
stay consistent and changeable in one place.

## Surfaces

| Surface | Function | Example output |
|---|---|---|
| Transition suggestion | `transition_suggestion_text` | `"Was this Coding? 12:00–12:47"` |
| Live status | `live_status_text` | `"Now: Coding"` |
| Gap-fill prompt | `gap_fill_prompt` | `"You have 2h 30m unlabeled. Review?"` |
| Gap-fill detail | `gap_fill_detail` | `"Review unlabeled: 9:00–11:30"` |
| Activity source setup title | `activity_source_setup_title` | `"Activity source unavailable"` |
| Activity source setup body | `activity_source_setup_message` | `"Manual labeling still works, but activity summaries..."` |
| Activity source setup steps | `activity_source_setup_steps` | `["Install and start ActivityWatch.", ...]` |
| Activity source help URL | `activity_source_setup_help_url` | `"https://activitywatch.net/"` |

## Design conventions (Decision 6)

- **Action-oriented framing** with concrete time ranges for transition suggestions.
- Transition suggestion time ranges are rendered in the user's **local timezone** for display, while structured interval fields remain UTC elsewhere in the UI/API.
- **Present-tense statement** for live status, no time range needed.
- **No hedging language** ("We think you might have been...").
- **No numeric confidence** on live status or transition surfaces.
- Activity-source setup guidance is also centralized here so the backend can
  return the same copy to every frontend surface.

## API

::: taskclf.ui.copy
