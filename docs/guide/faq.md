# FAQ

## `sieval dataset download`

### HF progress bar interleaves with sieval logs

HF's tqdm and loguru both write to stderr. Workarounds:

- `HF_HUB_DISABLE_PROGRESS_BARS=1` — silent, but no byte-level progress
- Leave defaults — interleaving is cosmetic

### Batch download

One name, one `--domain`, or `--all`. Multi-name and repeated `--domain` not yet supported. Until then:

```bash
for n in aime_2024 math_500 mmlu; do sieval dataset download "$n"; done
```

### Format on disk + `path:` in YAML

`hf:` sources land at `$SIEVAL_DATA_DIR/<org>/<name>/` as plain files (`huggingface_hub` `local_dir=` mode — flat, not the hub-cache slug layout). `url:` sources land at `$SIEVAL_DATA_DIR/<dataset_name>/<basename>`, **not extracted** — `.gz` stays `.gz` (HF `load_dataset` reads it streaming).

YAML `path:` for `hf:`-scheme datasets stays as the bare repo_id; runtime resolves it to the on-disk path by string concat, so `load_dataset` takes the local-dir branch (no online/offline split). Existing `HF_HUB_CACHE` in the environment is left alone for `huggingface_hub`'s model / other consumers.

```yaml
datasets:
  aime_2024:                          # hf-scheme: bare repo_id
    class: AIME2024Dataset
    path: "HuggingFaceH4/aime_2024"

  drop:                               # url-scheme: directory
    class: DROPDataset
    path: "${SIEVAL_DATA_DIR}/drop"
```

### Upgrading from 0.5.0

0.5.0 staged `hf:` datasets at `$SIEVAL_DATA_DIR/hf/datasets--<org>--<name>/snapshots/<sha>/` (hub-cache slug layout). The current release moves them to `$SIEVAL_DATA_DIR/<org>/<name>/` (flat). Existing downloads under the old path are not picked up — re-stage with:

```bash
rm -rf "$SIEVAL_DATA_DIR/hf" && sieval dataset download --all
```
