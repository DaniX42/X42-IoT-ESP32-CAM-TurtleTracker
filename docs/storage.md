# Storage

The default persistence layer is SQLite at `data/turtle_tracker.db`.

## `positions`

Stores timestamp, enclosure coordinates (`x`, `y`), house state, speed, and detector confidence. Timestamps are stored as UTC ISO-8601 strings and the timestamp column is indexed.

## `events`

Stores timestamped domain events with an event name and optional details. Planned event names include `entered_house`, `left_house`, and `motion_detected`.