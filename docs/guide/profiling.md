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
