# V5 operation manual

This directory records how to run the first V5 uppercase multi-style addition experiment.

## Resource rule

Do not run training or heavy recomputation on the login node. Request one GPU first:

```bash
salloc -N 1 --gres=gpu:1 --mem=32G
conda activate xuanjie
```

Release the interactive allocation when the runs finish:

```bash
exit
```

## Experiment

Goal: freeze a pretrained V3 uppercase encoder/VQ/decoder and train only a content-space ELPIS-style addition module on mixed-style uppercase letter triplets.

Defaults:

- V3 checkpoint: `../variance-versus-invariance/logs/uppercase_letters_run2_batch32_resume/cp_epoch304.pt`
- Number mapping: `0=A, 1=B, ..., 20=U`
- Ordered triplets: `(a,b,c)` where `a+b=c` and `0<=a,b,c<=20`
- Split: fixed 30% train / 70% eval by ordered triplet
- Train images: `../data/UppercaseLetters/train`
- Eval images: `../data/UppercaseLetters/test`

Self-supervised boundary:

- The training loss may use only the perceptual triplet images `(x_a, x_b, x_c)` and their frozen V3 content/VQ representations.
- Numeric/letter ids are allowed for constructing the synthetic perceptual triplet stream and for diagnostics.
- Numeric/letter ids must not define the loss target, a canonical target code, or extra same-class positives.
- `number_accuracy`, heatmaps, confusion plots, and V3 code-to-letter mappings are labelled probes, not training signals.

Main label-free config families:

- `configs/label_free_pca_d2_mse_ste.yaml`
- `configs/label_free_pca_d4_mse_ste.yaml`
- `configs/label_free_pca_d8_mse_ste.yaml`
- `configs/label_free_learned_d2_recon_mse_ste.yaml`
- `configs/label_free_learned_d4_recon_mse_ste.yaml`
- `configs/label_free_learned_d8_recon_mse_ste.yaml`
- `configs/label_free_vq_ce_hard.yaml`
- `configs/label_free_vq_ce_soft.yaml`

Config switches:

- `operation_space: identity|pca|learned` selects the content space used by the plus net.
- `operation_dim: 2|4|8|...` sets the PCA or learned bottleneck dimension.
- `projector_recon_loss_weight` anchors a learned projector with an autoencoder reconstruction loss.
- `prediction_mode: content|code_logits` selects MSE+STE content prediction or direct VQ-code logits.
- `target_vq_ce_loss_weight` enables CE against the V3 VQ index of the observed target image `x_c`.
- `target_vq_ce_mode: hard|soft` selects exact target-index CE or codebook-distance soft CE.
- `plus_hidden_dim`, `plus_hidden_layers`, `learning_rate`, and `max_steps` are the first knobs for bigger-net, LR, and longer overfit sweeps.
- `vis_sample_seed` controls reproducible random sampling for `triplet_grid_step_<N>.png`; if omitted, `seed` is used.

## Run Naming

Use this run-name format for new experiments:

```text
YYYYMMDD-HHMM__experiment_slug
```

Examples:

```text
20260623-0642__dvq_ratio01_h4096x4_c01
20260623-0715__hard_ce_lr3e5_resume
```

Rationale:

- Lexicographic order is chronological order.
- The minute-level timestamp avoids same-day collisions.
- `-` and `_` are easy to handle in shell scripts.
- Keep the slug short but include the method and the most important hyperparameters.

## Reading The Visuals

- `train_addition_heatmap.png` contains only pairs present in the train split.
- `eval_addition_heatmap.png` contains only pairs present in the eval/test split.
- Blank heatmap cells are absent pairs, either because `a+b>20` or because that pair is not in the split.
- A colored heatmap cell is the per-pair diagnostic number accuracy.
- With the current deterministic eval, each pair is evaluated once, so colored cells are usually `0` or `1`. Intermediate values require repeated eval samples per pair.
- `number_accuracy` maps a predicted VQ atom through a labelled code-to-number diagnostic probe before comparison.
- `code_accuracy` requires exact equality between the predicted VQ atom id and the observed target image's VQ atom id.
- `triplet_grid_step_<N>.png` samples random eval/test triplets, not train triplets.

## Commands

From an allocated GPU shell:

```bash
cd /home/xuanjie.liu/Projects/v5
python -m src.train --config configs/label_free_pca_d2_mse_ste.yaml
python -m src.train --config configs/label_free_vq_ce_hard.yaml
```

Short GPU smoke:

```bash
cd /home/xuanjie.liu/Projects/v5
python -m src.train --config configs/label_free_pca_d2_mse_ste.yaml --run_name smoke_pca_d2 --max_steps 5 --eval_interval 5 --batch_size 4
python -m src.train --config configs/label_free_vq_ce_hard.yaml --run_name smoke_vq_ce --max_steps 5 --eval_interval 5 --batch_size 4
```

## Outputs

Each run writes to:

```text
v5/runs/<run_name>/
```

Key files:

- `config.yaml`
- `split.json`
- `number_letter_map.json`
- `v3_codebook_mapping.json`
- `metrics.csv`
- `metrics.jsonl`
- `latest.pt`
- `best_1.pt`
- `vis/metrics.png`
- `vis/train_addition_heatmap.png`
- `vis/eval_addition_heatmap.png`
- `vis/eval_confusion.png`
- `vis/triplet_grid_step_<N>.png`

## Notes

The V3 codebook atom id is not assumed to match the letter id. The run estimates an atom-to-letter majority mapping from uppercase train pages and saves it in `v3_codebook_mapping.json`.

## Artifact Policy

- Do not commit `runs/` or checkpoint files.
- Checkpoints are local artifacts and are ignored by git.
- Preserve lightweight experiment evidence in `experiment_records/` and in the experiment report files.
- Generate/update the lightweight run manifest before deleting local run artifacts:

```bash
python scripts/collect_run_manifest.py
```

- Preview cleanup first, then apply it:

```bash
python scripts/clean_runs.py
python scripts/clean_runs.py --apply
```
