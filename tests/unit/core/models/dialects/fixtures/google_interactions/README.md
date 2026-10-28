# Google Interactions response fixtures

`end-of-file-fixer` skips the `.sse` files here: their final blank lines are
SSE frame separators.

Live captures, from `gemini-3.6-flash` on `POST /v1beta/interactions` with
`store: false` and `thinking_summaries: auto`, October 27–28, 2026:

- `stream_thought_summary_completed.sse`, `stream_thought_summary_incomplete.sse`:
  two thought-summary items each, text and whitespace verbatim (budgets 2048
  and 1024). `ReasoningOutput.text` concatenates them without separators, as
  Google's [streaming example](https://ai.google.dev/gemini-api/docs/streaming#streaming-with-thinking)
  prints them.
- `stream_signature_only_completed.sse`, `stream_multiple_functions_signature_only.sse`,
  `stream_url_context_signature_only.sse`, `stream_google_search_signature_only.sse`:
  a thought signature with no summary delta, which `summary: auto` allows.
- `function_choice_none_provider_violation.json`: after an error function
  result, the provider emitted a call despite `tool_choice: none`; the dialect
  must reject it.

All live terminal events omit `steps`. Signatures and call IDs are placeholders,
and no capture contains request headers, keys or real interaction IDs. The other
fixtures, and tests that derive a terminal `steps` snapshot from these, are
synthetic.
