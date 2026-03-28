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

## Design conventions (Decision 6)

- **Action-oriented framing** with concrete time ranges for transition suggestions.
- **Present-tense statement** for live status, no time range needed.
- **No hedging language** ("We think you might have been...").
- **No numeric confidence** on live status or transition surfaces.

## API

::: taskclf.ui.copy
