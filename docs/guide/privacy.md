# Privacy & Data Handling Policy v1

Version: 1.0  
Status: Stable  
Last Updated: 2026-02-23  

This document defines mandatory privacy constraints for data collection,
storage, training, and inference.

All components MUST comply.

---

# 1. Privacy Principles

The system is designed to:

- Collect the minimum data required
- Avoid raw content storage
- Prevent reconstruction of user activity content
- Support on-device operation by default
- Avoid centralized raw behavioral logging

Raw text, keystrokes, URLs, and document content MUST NEVER be stored.

---

# 2. Prohibited Data

The following data MUST NOT be stored, transmitted, or logged:

- Raw keystrokes
- Clipboard contents
- Raw window titles
- Full URLs
- File contents
- Email/chat message contents
- Screenshots
- Audio/video streams

No exceptions.

---

# 3. Allowed Data (Derived Signals Only)

The system may store:

## 3.1 Interaction Metrics (Aggregated Only)

- keys_per_min
- backspace_ratio
- shortcut_rate
- clicks_per_min
- scroll_events_per_min
- mouse_distance
- app_switch_count_last_5m

These are numeric aggregates only.

Raw event streams must be discarded after window aggregation.

---

## 3.2 Application Identity

Allowed:

- `app_id` (process name or bundle identifier)
- `is_browser`
- `is_editor`
- `is_terminal`

Not allowed:

- Raw window title
- Full executable path (unless sanitized)

---

## 3.3 Window Title Handling

If window titles are used:

- They MUST be hashed
- Hash MUST be salted per user
- Salt MUST be stored locally
- Salt MUST NOT leave device

Recommended:

```

window_title_hash = SHA256(user_salt + raw_title)

```

Raw titles must be immediately discarded after hashing.

---

## 3.4 Browser URL Handling

If browser signals are added:

Allowed:

- Domain category
- eTLD+1 (e.g., example.com)
- Predefined domain classification label

Not allowed:

- Full URL path
- Query parameters
- Fragment identifiers

---

# 4. Data Storage Model

## 4.1 Default Mode

All raw event logs:
- Stored locally
- Processed into windows
- Immediately discarded

Only aggregated window features persist.

---

## 4.2 Remote Training Mode (Optional)

If centralized training is used:

Only the following may be transmitted:

- Aggregated window feature vectors
- Core label
- user_id (hashed)
- schema version

Transmission must use encrypted channel.

No raw event streams may be transmitted.

---

# 5. User Identifier Policy

`user_id` must be:

- Random UUID
- Not email
- Not name
- Not machine serial

If transmitted:
- It must be hashed or anonymized.

---

# 6. Data Retention Policy

Raw event data:
- Retention: < 5 minutes
- Must be discarded after aggregation

Aggregated window data:
- Retention: user configurable
- Default: unlimited (local only)

Manual labels:
- Retained unless user deletes

---

# 7. Model Artifact Privacy

Model artifacts:

- Must not contain raw titles or URLs
- May contain hashed categorical features
- May contain aggregated statistics

If exporting model externally:
- Ensure no reversible hashes
- No embedded salts

---

# 8. Active Learning Prompts

When prompting user to label:

UI must display:
- Time range
- Application names (sanitized)
- Aggregated stats

UI must NOT display:
- Raw typed text
- Raw content previews

---

# 9. Device Boundary

If multiple devices exist:

Each device may:
- Collect raw events locally
- Aggregate locally
- Send only aggregated windows to central store

Raw events must never cross device boundary.

---

# 10. Differential Privacy (Optional Future Enhancement)

If deploying across many users:

Consider:
- Adding noise to non-critical aggregate metrics
- Removing low-frequency categorical values
- Clipping extreme interaction values

Not required for v1.

---

# 11. Security Requirements

- Local storage encrypted at rest (recommended)
- Encrypted transport if remote
- No plaintext logs containing user behavior
- Hash salts stored securely

---

# 12. Compliance Boundary

This system is designed to avoid:

- Content surveillance
- Behavioral reconstruction
- Sensitive information inference

The classifier operates purely on behavioral metadata,
not semantic content.

---

# 13. Versioning

Changing any of the following requires version bump:

- Allowed data types
- Hashing policy
- Storage boundary rules
- Transmission rules
