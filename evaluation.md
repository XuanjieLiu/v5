# V5 evaluation guide

This guide explains how V5 metrics and visualizations are computed, how to read them, and what ideal behavior should look like. All metrics here are diagnostics. The training objective remains label-free unless explicitly stated otherwise in a config.

## Dataset Splits

V5 uses ordered addition triples:

```text
(a, b, c), where a + b = c and 0 <= a,b,c <= 20
```

There are 231 valid ordered pairs/triples:

```text
21 + 20 + ... + 1 = 231
```

The uppercase mapping is:

```text
0=A, 1=B, ..., 20=U
```

This mapping is used to construct perceptual triplets from glyph images. It is not a training target. A training item contains three images:

```text
(x_a, x_b, x_c)
```

The model sees frozen V3 representations of these images.

Two split modes matter:

- `triplet_split_mode: all`: train and eval both contain all 231 numeric pairs, but train images come from `letter_train_dir` and eval images come from `letter_eval_dir`.
- `triplet_split_mode: random`: ordered pairs are split by `train_ratio`. With `train_ratio: 0.3`, train has 69 pairs and eval has 162 pairs.

## Loss Metrics

### `loss`

What it measures:

- The weighted training/eval objective assembled from the active config.
- It may include MSE, VQ CE, projector reconstruction, symmetry, and commit losses.

How to read it:

- Lower is better within the same config.
- Do not compare absolute `loss` values across configs with different loss weights or prediction modes.

Ideal:

- Decreases steadily and correlates with rising `code_accuracy` and `number_accuracy`.

### `plus_loss`

What it measures:

- `MSE(pred_vq, target_content_vq)`.
- `pred_vq` is the predicted content after quantization/STE or code-logit soft/hard projection.
- `target_content_vq` is the frozen V3 quantized content of the observed target image `x_c`.

How to read it:

- Useful for MSE+STE configs.
- In `prediction_mode: code_logits`, it is a diagnostic and may not be the optimized loss.

Ideal:

- Low and decreasing, with accuracy also improving.

### `raw_plus_loss`

What it measures:

- Optional MSE before VQ quantization, when the raw prediction shape matches content space.

How to read it:

- Only meaningful when `raw_plus_loss_weight > 0`.

Ideal:

- Should decrease if enabled, but it is secondary to target VQ/code behavior.

### `target_vq_ce_loss`

What it measures:

- Cross-entropy against the V3 VQ index of the observed target image `x_c`.
- In `hard` mode, the target is the exact VQ atom id of `x_c`.
- In `soft` mode, the target distribution is derived from distances between `x_c` content and all VQ codebook atoms.

Why it is label-free:

- The target comes from the perceptual target image `x_c`, not from the symbolic number or letter label.

How to read it:

- Lower is better.
- This is currently the strongest legal signal in short runs.

Ideal:

- CE falls while `code_accuracy` and `number_accuracy` rise.

### `projector_recon_loss`

What it measures:

- Reconstruction MSE for `operation_space: learned`.
- The learned projector encodes 512-d V3 content to a low-dimensional operation space and decodes it back.

How to read it:

- Confirms the learned bottleneck has not collapsed.
- A low reconstruction loss alone does not mean addition is learned.

Ideal:

- Drops quickly and stays low, while `plus_loss` and accuracies also improve.

### `symmetry_loss`

What it measures:

- Optional ELPIS-style content consistency over compositions such as `(a+b)+c` and `(a+c)+b`.

How to read it:

- Only meaningful when `use_symmetry: true`.
- It regularizes algebraic structure but is not itself a labelled accuracy.

Ideal:

- Decreases without hurting direct addition accuracy.

### `commit_loss`

What it measures:

- VQ commit loss from the predicted content quantization path.

How to read it:

- Mostly a stability diagnostic for content prediction modes.

Ideal:

- Stable and not dominating the total objective.

## Accuracy Metrics

### `code_accuracy`

What it measures:

- Exact VQ atom match:

```text
predicted VQ atom id == V3 VQ atom id of target image x_c
```

How to read it:

- This is the stricter diagnostic.
- It can be low even if the predicted atom corresponds to the right symbolic number, because V3 may use multiple atoms for one number/style family.

Ideal:

- Approaches the V3 oracle exact-code ceiling.
- Earlier diagnostics found the exact-code oracle is below perfect because V3 has aliasing/style variation.

### `number_accuracy`

What it measures:

- Maps the predicted VQ atom through a labelled majority-code probe, then compares the mapped number with `c`.

Important:

- This is a labelled diagnostic probe only.
- It is not used in the training loss.

How to read it:

- Less strict than `code_accuracy`.
- It can be higher than `code_accuracy` when several VQ atoms map to the same number.
- It can also be misleading if the V3 code-to-number mapping is ambiguous.

Ideal:

- Approaches the V3 oracle number ceiling.
- Earlier diagnostics found the number oracle can be much higher than exact-code oracle.

## `metrics.csv` and `metrics.jsonl`

What they contain:

- One row/object per eval event.
- Each row has train and eval versions of the loss and accuracy metrics.

How to read them:

- Use `eval_*` metrics for held-out style/image generalization.
- Use `train_*` metrics to diagnose optimization on the training split.
- A good run should improve both, with eval not lagging too far behind train.

Ideal:

- `eval_target_vq_ce_loss` or `eval_plus_loss` decreases.
- `eval_code_accuracy` and `eval_number_accuracy` rise.
- Train/eval gaps stay small unless intentionally overfitting.

## `metrics.png`

What it shows:

- Left panel: loss curves from `metrics.csv`.
- Right panel: diagnostic accuracy curves.

How to read it:

- Compare curves within a single run.
- Do not compare absolute loss levels across configs unless their loss definitions match.
- `number_accuracy` and `code_accuracy` answer different questions:
  - `code_accuracy`: exact VQ target atom match.
  - `number_accuracy`: symbolic number after a diagnostic VQ-to-number mapping.

Ideal:

- Loss decreases smoothly.
- Both eval accuracies trend upward.
- `number_accuracy` may be above `code_accuracy`, but both should improve.

## `train_addition_heatmap.png`

What it shows:

- Per-pair diagnostic number accuracy for pairs in the train split.
- X-axis is `b`; Y-axis is `a`.

Blank cells:

- Pair is absent from the train split, or invalid because `a+b>20`.

Why cells are often 0 or 1:

- Current deterministic eval samples one perceptual example per present pair.
- With one example per pair, per-cell accuracy is binary.
- Intermediate values require repeated eval samples per pair.

Ideal:

- Present valid train cells become bright/yellow.
- Blank cells remain blank.

## `eval_addition_heatmap.png`

What it shows:

- Per-pair diagnostic number accuracy for pairs in the eval/test split.
- X-axis is `b`; Y-axis is `a`.

Blank cells:

- Pair is absent from the eval split, or invalid because `a+b>20`.

How to read it:

- This is the best visual summary of which held-out pairs/styles are working.
- In `triplet_split_mode: all`, all 231 valid pairs are present, but evaluated on held-out image pages/styles.
- In `triplet_split_mode: random`, only eval-split pairs are shown.

Ideal:

- Present valid eval cells become bright/yellow.
- Generalization failures appear as dark present cells, not blank cells.

## `eval_confusion.png`

What it shows:

- Rows are target numbers.
- Columns are predicted numbers after the diagnostic VQ-to-number mapping.

How to read it:

- Diagonal mass means correct symbolic addition.
- Vertical or horizontal bands indicate systematic prediction collapse.
- Off-diagonal patterns show biased errors, such as consistently underpredicting or overpredicting.

Ideal:

- Bright diagonal, little off-diagonal mass.

## `triplet_grid_step_<N>.png`

What it shows:

- Random eval/test triplets at a given step.
- Columns are:

```text
a image | b image | target c image | predicted c image
```

Sampling:

- The grid samples from the eval/test loader.
- It uses `vis_sample_seed + step * 1009` for reproducible random samples.
- It does not use the first sorted batch, so it should not always start with `A+A=A`.

How to read it:

- Use it for qualitative inspection of decoded predictions.
- It is especially useful for spotting collapse, blurry prototypes, style leakage, or repeated predictions.

Ideal:

- The predicted `c` image should have the target content identity.
- Style may follow the target-style decode path used for visualization.

## Checkpoint Files

### `latest.pt`

What it is:

- The latest checkpoint at the most recent eval point.

### `best_1.pt`, `best_2.pt`, `best_3.pt`

What they are:

- Current top-k checkpoints sorted by eval loss.

Important:

- Best eval loss is not always best `number_accuracy`.
- For scientific comparison, inspect `metrics.csv` and report both loss and diagnostic accuracies.

## Current Interpretation Rules

- Treat `number_accuracy` and all number heatmaps as labelled probes, not training evidence by themselves.
- A result is most convincing when:
  - the training loss is label-free,
  - eval `code_accuracy` rises,
  - eval `number_accuracy` rises,
  - heatmaps brighten on present pairs,
  - triplet grids stop showing collapse or repeated prototypes.
- If `loss` improves but both accuracies stay near chance, the model is optimizing a weak surrogate without learning useful addition.
