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

With `record_meta` on (the default), `profile.json` is built when the task
finishes from every sample's recorded `stage_meta`: stage timings and token
usage, resumed samples included, whatever `profile_stages` and `profile_usage`
say. `profile_io` adds flush and load timings for this run only.
`progress.json` reads `finished: true` once the task has finished, on every
exit path, a reused `report.json` included.
