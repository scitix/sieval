# Profiling & Observability

Enable profiling via `TaskRunnerConfig`:

```python
config = TaskRunnerConfig(
    profile_stages=True,   # per-stage timing
    profile_io=True,       # flush/load timing
    profile_usage=True,    # token counts (default: True)
    show_progress=True,    # progress bar
    dump_progress=True,    # save progress.json
)
```

With `record_meta` on (the default), `profile.json` covers every sample's
recorded stage timings and token usage, resumed samples included, regardless of
`profile_stages` / `profile_usage`. `profile_io` timings cover this run only.
