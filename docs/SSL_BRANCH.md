# Self-supervised pretraining lives on the `ssl` branch

SimMIM and JEPA pretraining were split out of `main` so the supervised
ablation framework is one task with one training loop. Nothing was thrown
away — the `ssl` branch carries the whole thing, and it was cut *after* the
LayerNorm-only, MegaBlocks-removal, fail-fast-loss and real-RoPE work, so it
inherits all of it.

```bash
git checkout ssl
python tests/run_all.py                                   # same gate, still green
python train.py --task ssl --dataset pass --dry-run       # SimMIM pretraining
```

## What is over there and not here

| | |
|---|---|
| `pvt_moe/ssl/` | `simmim.py` (`LitSimMIM`, the default method), `jepa.py` (`LitJEPA`), `backbone.py`, `masking.py`, `predictor.py`, `diagnostics.py` |
| config | the `task` axis (`supervised` \| `ssl`), the whole `cfg["ssl"]` subtree, `ssl.method` / `mask_space` / `mask_ratio`, the `ssl_finetune` recipe, the unlabelled **PASS** dataset |
| CLI | `--task ssl`, `--ssl-method`, `--mask-space`, `--mask-ratio`, `--mask-patch-size` |
| engine | `build_ssl_trainer`, the SSL arms of `run_identity` and the results record |
| data | `build_ssl_transform` and the unlabelled-corpus path through `HFImageDataset` |
| backbone | `forward_features(stage1_token_mask=…, mask_token=…)` — SimMIM-style masking after the stage-1 embed |
| docs | `SIMMIM_GUIDE.md`, `JEPA_GUIDE.md` |
| notebooks | `03_ssl_pretrain.ipynb` |
| configs | `ssl_01`…`ssl_04` |
| tests | `test_simmim.py`, `test_masking.py`, `test_ssl_init.py`, `test_cli_ssl.py`, `test_pass_dataset.py` |

## What stayed on `main`, and why

- **`pvt_moe/eval/`** — k-NN, linear probe and the low-shot subset builder.
  They read as SSL tooling but are not: `eval/lowshot.py` backs
  `dataset.subset_file` for any dataset and is imported by `download_data.py`,
  and `eval/runner.py` does supervised validation top-1.
- **`mode: "warm_start"`** — renamed from `ssl_init`, which was never
  SSL-only: the `downstream` recipe sets it, so it runs on every small-dataset
  fine-tune. It means "warm start from a local checkpoint", and the old name
  was actively misleading once SSL left.
- **`cfg["chain"]`** — still records the stages that produced a run, now
  `hf_finetune@imagenet-1k_r224 -> downstream@eurosat_r224`.

## Bringing the two back together

The branches share history up to the cut, so `git merge` works normally. The
SSL side will need `task`, `cfg["ssl"]` and the `ssl_finetune` recipe put back
into `config/`, and `mode: "ssl_init"` reconciled with `warm_start`.
