# Examples Gallery

Ready-to-run configs for common evaluation scenarios. Pick the file that
matches what you're trying to do, copy it, edit the marked fields, and run
`sieval eval <your-copy>.yaml`.

## Scenario-indexed (novice path)

| File | When to pick |
| --- | --- |
| [quickstart.yaml](quickstart.yaml) | Single task + single model + 5 samples — smoke test your install |
| [leaderboard-math-sft.yaml](leaderboard-math-sft.yaml) | Math SFT leaderboard — multiple math tasks against one or more models |
| [infer-recipe-override.yaml](infer-recipe-override.yaml) | Pin a specific inference recipe or override engine args |

## Hardware-indexed (reference configs)

See [hardware/](hardware/) for known-working configs pinned to specific
hardware × model combinations. Copy when you want a starting point that's
proven to run on your box.

## Discovery

- `sieval dataset list` — show registered datasets, licenses, and download status
- `sieval task list` — show registered tasks with eval mode, n_shot, and dependencies
- `sieval dataset show <name>` — dataset detail + tasks using it
- `sieval task show <name>` — task detail + reference implementation

## Notes

- `class:` fields in these YAMLs refer to Task/Dataset **Python class names**
  (e.g. `AIME2024ZeroShotGenTask`), not the registered meta names you see
  in `sieval task list`.
- When run via `sieval run <config.yaml>`, `models.*.args.name`, `api_base`,
  and `api_key` are auto-injected from the launched inference service. When
  run via `sieval eval <config.yaml>` against an external endpoint, provide
  them explicitly under `models.*.args`.
