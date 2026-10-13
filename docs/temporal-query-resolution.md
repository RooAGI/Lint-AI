# Temporal query resolution

Temporal language is extracted during query analysis and resolved by
`PreparedQuery` against either the query's explicit reference date or the
current date. `PreparedQuery::temporal_context()` is the shared handoff to
single-index ranking and segmented routing.

For example, “during the past month” is recognized as an explicit temporal
range. With a reference date of `2023-07-01`, it resolves to the inclusive
interval `2023-06-01` through `2023-07-01`. Month and year offsets use calendar
arithmetic; day and week offsets use day counts. The range is passed as
`starts_from` and `ends_at`, and the same bounds become the segmented routing
window. A resolved closed interval is a hard constraint: documents with
missing, unparseable, or out-of-range timestamps are removed before lexical
and hybrid candidates are collected. This prevents out-of-window candidates
from crowding the ranking before temporal scoring. Explicit ranges also suppress
the legacy point-target parser and keep sequence reranking active.

Point expressions such as “last Tuesday” continue to produce an anchor and a
seven-day routing window around it. Session-provided absolute anchors retain
their existing path.

Timestamps describe when a memory was recorded, not necessarily when an event
mentioned in it happened. Hard filtering therefore treats the recording date as
the available temporal proxy. Point expressions without a closed interval keep
their existing proximity-scoring behavior; they do not activate this hard
filter.
