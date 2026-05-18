# Leaderboards

Leaderboard configuration YAMLs consumed by `sieval eval` and `sieval leaderboard report`.

## Layout

```text
leaderboards/
├── <name>.yaml              # one leaderboard = one YAML
└── README.md                # this file
```

## Two flavors

- **Free-form** — `models` × `tasks` is ad-hoc, no reference baseline. Example: [sft_fast_202511.yaml](sft_fast_202511.yaml).
- **TR-aligned** — carries a top-level `alignment: { card: <path>/<stage>.md }` block. The card is the reproducibility contract; `sieval leaderboard report` joins observed runs against its reference scores. No alignment cards ship in this release; users author their own under the convention path `leaderboards/alignment/<tr-slug>/<stage>.md`.

Both flavors share the same loader and runner. For the YAML schema, read the existing YAML in this directory or `sieval eval --help`.
