# V5 label-free addition experiment report

## Scope

The active goal is to make V5 learn addition without symbolic labels in the training loss. Training targets may come from perceptual triplets `(x_a, x_b, x_c)` and frozen V3/ELPIS representations only. Numeric or letter ids are allowed for dataset construction and diagnostic probes, but not for defining canonical target codes or same-class positive sets.

## Current method priority

1. Label-free PCA `d=2/4/8` with MSE+STE.
2. Learned projector with reconstruction anchor and MSE+STE.
3. Target-image VQ index cross-entropy.
4. Learning-rate and longer overfit sweeps.
5. Bigger plus-net.
6. Direct VQ-Plus: S3Plus-style V3 content VQ addition with MSE+STE.

## Symmetry/eval caveat

- All current `label_free_*` best runs use `use_symmetry: false`.
- The strongest hard-CE runs use `triplet_split_mode: all`, so eval tests held-out pages/styles for all arithmetic pairs, not unseen-pair addition generalization.
- In no-symmetry settings, train addition accuracy should be reported alongside eval accuracy. A high eval score under `triplet_split_mode: all` is not evidence of ELPIS-style symmetry generalization.

## Code/config changes

- Added explicit label-free configs:
  - `configs/label_free_direct_vq_plus_mse_ste.yaml`
  - `configs/label_free_pca_d2_mse_ste.yaml`
  - `configs/label_free_pca_d4_mse_ste.yaml`
  - `configs/label_free_pca_d8_mse_ste.yaml`
  - `configs/label_free_learned_d2_recon_mse_ste.yaml`
  - `configs/label_free_learned_d4_recon_mse_ste.yaml`
  - `configs/label_free_learned_d8_recon_mse_ste.yaml`
  - `configs/label_free_vq_ce_hard.yaml`
  - `configs/label_free_vq_ce_soft.yaml`
- Removed the label-derived `multi_positive` target path from `src/data.py`, `src/train.py`, and `src/eval.py`.
- Documented the self-supervised boundary and config switches in `operation_manual/README.md`.
- Changed checkpoint retention for future runs from `save_top_k: 3` to `save_top_k: 1`, so each run keeps only `latest.pt` and `best_1.pt`.
- Added `--save_top_k` CLI override to `src/train.py`.
- Updated `TopKCheckpoints` to load an existing `best_1.pt` on resume and remove stale `best_*.pt` files above the configured `k`.
- Added CLI overrides for Direct VQ-Plus hyperparameter sweeps: learning rate, weight decay, plus hidden size/layers, train input mode, train ratio, split mode, plus/raw loss weights, and commit weight.
- Fixed Direct VQ-Plus commitment loss: because frozen V3 is held in eval mode, the library VQ commitment loss was zero. `src/plus_model.py` now explicitly computes `MSE(raw_content, quantized_hard.detach())`.

## Validation log

### 2026-06-21 code/config smoke

Checks:

- `grep -RIn "multi_positive\|target_positive" src configs operation_manual README.md` found no remaining matches.
- `python -m py_compile src/data.py src/eval.py src/plus_model.py src/train.py src/visualize.py src/analyze_v3_codes.py` passed.
- All `configs/label_free_*.yaml` files loaded and exposed the expected `operation_space`, `operation_dim`, `prediction_mode`, and `target_vq_ce_mode`.

GPU 1-step smoke:

| run | config path | result |
| --- | --- | --- |
| `smoke_label_free_pca_d2_check` | `configs/label_free_pca_d2_mse_ste.yaml` | step 1 completed, eval number acc 0.0260, eval code acc 0.0260 |
| `smoke_label_free_learned_d2_check` | `configs/label_free_learned_d2_recon_mse_ste.yaml` | step 1 completed, eval number acc 0.0390, eval code acc 0.0303 |
| `smoke_label_free_vq_ce_check` | `configs/label_free_vq_ce_hard.yaml` | step 1 completed, eval number acc 0.0823, eval code acc 0.0736 |

Reflection:

- The three priority code paths run end-to-end on CUDA.
- The 1-step scores are only plumbing checks, not learning evidence.
- Direct VQ CE starts higher than MSE+STE on labelled diagnostic probes, consistent with earlier runs where CE was the strongest legal branch.

### 2026-06-21 visualization fix smoke

Change:

- Heatmaps now use split-specific `heatmap_total`; absent cells are blank.
- Heatmap titles include the number of present pairs.
- Triplet grids now randomly sample eval/test triplets with a reproducible seed instead of taking the first sorted batch.

Validation run:

- `smoke_vis_random_split_check`

Observed artifacts:

- `vis/train_addition_heatmap.png` title shows `69 pairs; blank = absent`.
- `vis/eval_addition_heatmap.png` title shows `162 pairs; blank = absent`.
- `vis/triplet_grid_step_1.png` title shows `Random eval/test triplets at step 1`, and the sampled triples are not the fixed `A+A=A` prefix sequence.

Reflection:

- The heatmap now distinguishes wrong present pairs from pairs that are not part of the split.
- The binary-looking cells are expected for current deterministic eval because each present pair has one sampled example.
- Random triplet grids should make qualitative failures easier to inspect across the full eval/test split.

## Dataset and diagnostic semantics

Dataset composition:

- V5 uses ordered addition triples `(a,b,c)` with `a+b=c`, `0<=a,b,c<=20`.
- This yields `21+20+...+1 = 231` valid ordered pairs/triples.
- A number maps to an uppercase glyph only for data construction: `0=A, 1=B, ..., 20=U`.
- Each dataset item samples three perceptual glyph images `(x_a, x_b, x_c)` from a page directory. Training loss sees V3 content/VQ representations of these images, not the symbolic number as a target.
- In `triplet_split_mode: all`, train and eval contain all 231 numeric pairs, but train images come from `../data/UppercaseLetters/train` and eval images from `../data/UppercaseLetters/test`.
- In `triplet_split_mode: random`, the ordered pairs are split by `train_ratio`; with `train_ratio: 0.3`, that is 69 train pairs and 162 eval pairs.

Heatmap semantics:

- `train_addition_heatmap.png` is computed only from the train split loader.
- `eval_addition_heatmap.png` is computed only from the eval/test split loader.
- Blank cells are absent pairs: either invalid pairs where `a+b>20`, or pairs not present in that split.
- Cells are often exactly `0` or `1` because the current deterministic eval samples one perceptual triplet per valid ordered pair. With one sample per cell, per-cell accuracy is binary. Intermediate values require repeated eval samples per pair, e.g. multiple deterministic style/page samples per `(a,b,c)`.

Accuracy semantics:

- `code_accuracy` is exact VQ-code accuracy: predicted VQ atom id must equal the VQ atom id of the observed target image `x_c`.
- `number_accuracy` maps predicted VQ atom id through a labelled majority-code probe, then compares the resulting number with `c`.
- `code_accuracy` is stricter and sensitive to V3 code aliasing/style variation.
- `number_accuracy` can be correct even when the exact target VQ atom differs, if both atoms map to the same number under the diagnostic probe.
- Both are diagnostics only; neither is used by the training loss.

Triplet grid semantics:

- `triplet_grid_step_<N>.png` is generated from the eval/test loader.
- Before 2026-06-21, it used the first deterministic batch, so images often started with `A+A=A`.
- It now samples random eval/test triplets using `vis_sample_seed + step * 1009`, while staying reproducible for a given config and step.

## Experiment log

### 2026-06-21 priority 1: PCA d=2/4/8 + MSE+STE

Runs:

- `lf_pca_d2_mse_ste_200_20260621`
- `lf_pca_d4_mse_ste_200_20260621`
- `lf_pca_d8_mse_ste_200_20260621`

Status:

- Completed 200-step sweep with eval at steps 1, 100, and 200.

Results:

| run | best/current evidence | interpretation |
| --- | --- | --- |
| `lf_pca_d2_mse_ste_200_20260621` | final/best step 200, eval number acc 0.0476, eval code acc 0.0476, eval loss 7.9797 | weak signal only |
| `lf_pca_d4_mse_ste_200_20260621` | best step 100, eval number acc 0.0346, eval code acc 0.0346; final step 200 loss 7.3927 | loss falls but symbolic probe does not improve |
| `lf_pca_d8_mse_ste_200_20260621` | final/best step 200, eval number acc 0.0433, eval code acc 0.0433, eval loss 7.5132 | weak signal only |

Reflection:

- PCA bottlenecks do run cleanly, but short-run accuracy remains near chance.
- `d=4` and `d=8` reduce MSE more than `d=2`, yet this does not translate into target VQ/code accuracy.
- This supports the earlier suspicion that MSE+STE provides a weak optimization signal for landing on the nearest VQ code, even when the low-dimensional PCA space is label-free.
- Next step is priority 2: learned projector with reconstruction anchor. If that also fails in 200-step screening, move quickly to target-image VQ CE, which gave the strongest legal historical signal.

### 2026-06-21 priority 2: learned projector d=2/4/8 + recon anchor + MSE+STE

Runs:

- `lf_learned_d2_recon_mse_ste_200_20260621`
- `lf_learned_d4_recon_mse_ste_200_20260621`
- `lf_learned_d8_recon_mse_ste_200_20260621`

Status:

- Completed 200-step sweep with eval at steps 1, 100, and 200.

Results:

| run | best/current evidence | interpretation |
| --- | --- | --- |
| `lf_learned_d2_recon_mse_ste_200_20260621` | final step 200 eval number acc 0.0303, eval code acc 0.0346, eval plus loss 7.8579, eval recon loss 0.0341 | recon learns, addition does not |
| `lf_learned_d4_recon_mse_ste_200_20260621` | best step 100 eval number acc 0.0563; final step 200 eval number/code acc 0.0519, eval recon loss 0.0167 | best learned-projector short run, still weak |
| `lf_learned_d8_recon_mse_ste_200_20260621` | best step 100 eval code acc 0.0563; final step 200 eval number acc 0.0476, code acc 0.0433, eval recon loss 0.0155 | recon good, addition weak |

Reflection:

- The learned projector does not collapse: reconstruction loss drops from about 35 at step 1 to roughly 0.015-0.034 by step 200.
- The addition MSE remains high, around 7.86-7.94 at step 200, and labelled diagnostic accuracy remains near chance.
- This suggests the autoencoder anchor solves the "free coordinate collapse" concern, but not the harder problem of making MSE+STE hit the correct frozen V3 VQ basin.
- Priority 3 should now get focus: target-image VQ index CE is still label-free because the target index comes only from the observed perceptual target image `x_c`.

### 2026-06-21 priority 3: target-image VQ CE, 200 steps

Runs:

- `lf_vq_ce_hard_200_20260621`
- `lf_vq_ce_soft_200_20260621`

Status:

- Completed 200-step hard/soft CE screening.

Results:

| run | best/current evidence | interpretation |
| --- | --- | --- |
| `lf_vq_ce_hard_200_20260621` | final/best step 200 eval number acc 0.2771, eval code acc 0.2771, eval CE/loss 2.4382 | strongest legal signal so far |
| `lf_vq_ce_soft_200_20260621` | final/best step 200 eval number acc 0.1645, eval code acc 0.1732, eval CE/loss 2.8425 | learns, but slower/weaker than hard CE |

Reflection:

- Hard CE is clearly better than PCA+MSE+STE and learned-projector+MSE+STE in the same 200-step screen.
- This supports using target-image VQ index CE as the main branch for priority 4 LR/longer sweeps.
- Soft CE may be useful later as a fine-tuning or alias-aware regularizer, but it is not the best first path to "make addition learn".

### 2026-06-21 priority 4: hard VQ CE longer sweep, 1000 steps

Run:

- `lf_vq_ce_hard_1000_20260621`

Status:

- Completed 1000-step hard CE run with eval at steps 1, 200, 400, 600, 800, and 1000.

Results:

| step | eval CE/loss | eval code acc | eval number acc |
| --- | --- | --- | --- |
| 1 | 3.1218 | 0.0996 | 0.0823 |
| 200 | 2.4526 | 0.2771 | 0.2814 |
| 400 | 1.9344 | 0.3636 | 0.3463 |
| 600 | 1.7646 | 0.4026 | 0.4329 |
| 800 | 1.6981 | 0.3810 | 0.4286 |
| 1000 | 1.5214 | 0.4372 | 0.5022 |

Best:

- Best eval loss: step 1000, 1.5214.
- Best eval code accuracy: step 1000, 0.4372.
- Best eval number accuracy: step 1000, 0.5022.

Visual check:

- `vis/eval_addition_heatmap.png` shows all 231 valid pairs for `triplet_split_mode: all`; invalid cells are blank.
- `vis/triplet_grid_step_1000.png` uses random eval/test triplets. Predictions are no longer a single collapsed output, but there are still repeated prototype biases and many wrong targets.

Reflection:

- Hard target-image VQ CE is the first legal branch with strong early learning: eval number accuracy rises from 0.0823 to 0.5022 by 1000 steps.
- The run is still improving at step 1000 by loss and both accuracies, so a longer continuation is justified.
- The small dip at step 800 suggests checkpoint selection by both loss and accuracy should be monitored, not loss alone.
- Next priority should be a longer hard-CE sweep, likely 5k steps, then LR/bigger-net variants if improvement slows.

### 2026-06-22 priority 4: hard VQ CE continuation toward 5000 steps

Run:

- `lf_vq_ce_hard_resume5k_20260621`

Status:

- Resumed from `runs/lf_vq_ce_hard_1000_20260621/latest.pt` with optimizer state.
- The first continuation attempt was interrupted after step 1500.
- No background training process remained after interruption.

Partial results:

| step | eval CE/loss | eval code acc | eval number acc |
| --- | --- | --- | --- |
| 1001 | 1.5329 | 0.4762 | 0.5368 |
| 1500 | 1.3657 | 0.4805 | 0.5498 |
| 2000 | 1.2961 | 0.4978 | 0.5628 |
| 2500 | 1.2438 | 0.4978 | 0.5325 |
| 3000 | 1.1532 | 0.5455 | 0.6234 |
| 3500 | 1.0707 | 0.5758 | 0.6277 |
| 4000 | 1.0971 | 0.5671 | 0.6147 |
| 4500 | 1.0454 | 0.5455 | 0.6234 |
| 5000 | 1.0662 | 0.5628 | 0.6797 |

Reflection:

- The resumed optimizer state is valid: step 1001 is close to, and slightly better than, the previous step 1000 checkpoint.
- The hard-CE branch continued improving through 5000 steps in diagnostic number accuracy.
- Best diagnostic number accuracy is step 5000: `0.6797`.
- Best exact-code accuracy is step 3500: `0.5758`.
- Best eval loss is step 4500: `1.0454`.
- Best loss, best exact-code accuracy, and best number accuracy no longer select the same checkpoint, so later sweeps should track all three rather than relying on `best_*.pt` by loss alone.
- Next likely move: continue from step 5000 with a lower LR, or run a bigger plus-net variant if lower-LR continuation plateaus.

### 2026-06-22 priority 4: hard VQ CE lower-LR continuation

Run:

- `lf_vq_ce_hard_lr3e5_from5k_to6k_20260622`

Status:

- Resumed from `runs/lf_vq_ce_hard_resume5k_20260621/latest.pt`.
- Loaded optimizer state and changed learning rate from `1e-4` to `3e-5`.
- Completed step 5001 through 6000 with eval every 250 steps.

Results:

| step | eval CE/loss | eval code acc | eval number acc |
| --- | --- | --- | --- |
| 5001 | 1.0674 | 0.5628 | 0.6797 |
| 5250 | 1.0019 | 0.5931 | 0.6580 |
| 5500 | 0.9819 | 0.5844 | 0.6580 |
| 5750 | 1.0044 | 0.5628 | 0.6277 |
| 6000 | 1.0323 | 0.5584 | 0.6364 |

Best:

- Best diagnostic number accuracy remains the inherited step 5001/5000 level: `0.6797`.
- Best exact-code accuracy is step 5250: `0.5931`, a new overall best.
- Best eval loss is step 5500: `0.9819`, a new overall best.

Reflection:

- Lower LR improves exact VQ-code alignment and CE loss, but it does not improve the labelled number probe.
- The gap between code and number selection has become more pronounced, so future checkpoint selection should explicitly decide whether the target is exact VQ code or symbolic-number probe.
- Since number accuracy did not improve, the next useful branch is likely bigger plus-net capacity, or a lower-LR continuation selected from the best code/loss checkpoint if exact-code accuracy becomes the primary target.

### 2026-06-22 priority 5: bigger plus-net setup

Change:

- Added `configs/label_free_vq_ce_hard_big.yaml`.
- It keeps the same label-free hard target-image VQ CE objective.
- It increases plus-net capacity from hidden `2048 x 3` to hidden `4096 x 4`.

Validation run:

- `smoke_label_free_vq_ce_hard_big_check`

Result:

- Step 1 completed on GPU.
- Eval number acc `0.0563`.
- Eval code acc `0.0390`.
- Eval loss `3.4203`.

Reflection:

- The larger model fits in the requested 32G GPU allocation and the training path is valid.
- The smoke result is only a plumbing check; it is not learning evidence.
- Because lower LR improved code/loss but not number accuracy, a full big-net hard-CE run is now a reasonable next branch.

### 2026-06-22 priority 5: bigger plus-net 2000-step screen

Run:

- `lf_vq_ce_hard_big_2000_20260622`

Config:

- `configs/label_free_vq_ce_hard_big.yaml`
- `use_symmetry: false`
- `triplet_split_mode: all`
- hard target-image VQ CE
- plus-net hidden `4096 x 4`

Results:

| step | train add acc | eval CE/loss | eval code acc | eval number acc |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.0779 | 3.1499 | 0.1039 | 0.0779 |
| 500 | 0.5584 | 1.4233 | 0.4935 | 0.5455 |
| 1000 | 0.6407 | 1.1818 | 0.5541 | 0.6147 |
| 1500 | 0.6104 | 1.0713 | 0.5628 | 0.6104 |
| 2000 | 0.5844 | 1.1197 | 0.5195 | 0.5801 |

Best:

- Best eval number accuracy: step 1000, `0.6147`.
- Best eval code accuracy: step 1500, `0.5628`.
- Best eval loss: step 1500, `1.0713`.

Reflection:

- The bigger plus-net learns quickly early: by step 500 it reaches eval number acc `0.5455`.
- It does not beat the smaller long-run hard-CE model, whose best eval number acc is `0.6797`.
- The run peaks around step 1000/1500 and then drops by step 2000, so capacity alone is not an immediate fix.
- Stop here as requested; no new experiment launched.

### 2026-06-22 direct VQ-Plus setup

Method name:

- Direct VQ-Plus.

Config:

- `configs/label_free_direct_vq_plus_mse_ste.yaml`

Implementation alignment with `S3Plus/VQ/train.py`:

- S3Plus `VQVAE.plus(z_a, z_b)` concatenates two content VQ vectors, applies `plus_net`, then sends the continuous output through the VQ layer.
- S3Plus training can apply MSE to the quantized plus result (`e_ab`/`z_ab`) and adds the VQ commitment term.
- The V5 analogue uses frozen V3 content VQ vectors directly, predicts a raw 512d content vector, applies frozen V3 VQ with STE, and trains `MSE(pred_vq, target_vq)` plus a small commit loss.

Run:

- `lf_direct_vq_plus_mse_ste_5000_20260622`
- `use_symmetry: false`
- `triplet_split_mode: all`
- `plus_hidden_dim: 2048`
- `plus_hidden_layers: 3`
- `learning_rate: 1e-4`
- `max_steps: 5000`
- `eval_interval: 500`

Status:

- Initial run stopped at step 1500.
- Resumed from `runs/lf_direct_vq_plus_mse_ste_5000_20260622/latest.pt` and completed through step 5000.
- Checkpoint retention was changed during resume; the run directory now keeps only `latest.pt` and `best_1.pt`.

Results:

| step | train add acc | eval loss | eval code acc | eval number acc |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.0390 | 26.2669 | 0.0303 | 0.0390 |
| 500 | 0.0260 | 8.7792 | 0.0390 | 0.0260 |
| 1000 | 0.0606 | 9.3382 | 0.0476 | 0.0563 |
| 1500 | 0.0693 | 8.4127 | 0.0519 | 0.0649 |
| 1501 | 0.0303 | 17.1647 | 0.0216 | 0.0303 |
| 2000 | 0.0563 | 8.9938 | 0.0476 | 0.0519 |
| 2500 | 0.0649 | 8.4816 | 0.0433 | 0.0606 |
| 3000 | 0.0519 | 11.9386 | 0.0433 | 0.0563 |
| 3500 | 0.0866 | 8.3144 | 0.0779 | 0.0736 |
| 4000 | 0.0693 | 9.8470 | 0.0606 | 0.0649 |
| 4500 | 0.1082 | 6.3030 | 0.0866 | 0.0952 |
| 5000 | 0.1169 | 7.2470 | 0.1039 | 0.1039 |

Best:

- Best eval number accuracy: step 5000, `0.1039`.
- Best eval code accuracy: step 5000, `0.1039`.
- Best eval loss: step 4500, `6.3030`.

Reflection:

- This branch is a clean test of whether the original S3Plus-style MSE+STE objective works when applied directly to the frozen V3 content VQ space.
- It does not use numeric/letter labels in the loss and does not use target VQ CE.
- Direct VQ-Plus improves over the weakest PCA/learned-projector MSE+STE screens, but remains much weaker than target-image VQ CE.
- The main failure mode remains the same: MSE can reduce continuous distance, but it does not provide a strong enough signal to consistently land on the correct frozen V3 VQ atom.

### 2026-06-23 direct VQ-Plus hyperparameter sweep

Discovery:

- The original Direct VQ-Plus `pred_commit_loss_weight` was ineffective.
- Frozen V3 is kept in eval mode, and `vector_quantize_pytorch` computes commitment loss only when its module is in training mode.
- Therefore previous `train_commit_loss` and `eval_commit_loss` were zero even when `pred_commit_loss_weight: 0.1`.
- Fixed by explicitly computing `MSE(raw_content, quantized_hard.detach())` in `src/plus_model.py`.

Validation:

- A gradient sanity check after the fix gave nonzero commit loss and nonzero plus-net parameter gradients: `commit 8.0298`, `grad_sum 266.0512`.

Full-table fixed-code sweep:

| run | best train number acc | step | best eval number acc | step | best eval loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dvq_fixed_expcommit_c0_lr1e4_3000_20260623` | 0.0779 | 2500 | 0.0693 | 2500 | 7.9345 |
| `dvq_fixed_expcommit_c001_lr1e4_3000_20260623` | 0.0909 | 2500 | 0.0952 | 2500 | 5.9371 |
| `dvq_fixed_expcommit_c003_lr1e4_3000_20260623` | 0.1082 | 2500 | 0.0866 | 2500 | 5.7792 |
| `dvq_fixed_expcommit_c01_lr1e4_3000_20260623` | 0.1169 | 2500 | 0.0952 | 2500 | 5.5707 |
| `dvq_fixed_expcommit_c0_lr3e4_3000_20260623` | 0.1126 | 3000 | 0.0866 | 2500 | 8.0969 |

Raw-target sweep:

| run | best train number acc | step | best eval number acc | step | best eval loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dvq_fixed_raw1_vq0_c0_lr1e4_3000_20260623` | 0.1039 | 3000 | 0.0866 | 3000 | 5.4097 |
| `dvq_fixed_raw1_vq1_c0_lr1e4_3000_20260623` | 0.0952 | 2500 | 0.0866 | 2500 | 11.1050 |
| `dvq_fixed_raw1_vq0_c01_lr1e4_3000_20260623` | 0.0866 | 3000 | 0.0736 | 3000 | 5.3348 |
| `dvq_fixed_raw1_vq1_c01_lr1e4_3000_20260623` | 0.0866 | 2000 | 0.0779 | 1500 | 10.8878 |
| `dvq_fixed_raw1_vq0_c0_lr3e4_3000_20260623` | 0.1126 | 2500 | 0.0866 | 2500 | 5.1248 |

Train-ratio and capacity overfit sweep:

| run | train pairs | best train number acc | step | best train code acc | step | best eval number acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dvq_fixed_ratio01_h2048x3_c01_lr1e4_5000_20260623` | 23 | 0.6087 | 4000 | 0.7391 | 4000 | 0.0433 |
| `dvq_fixed_ratio01_h4096x4_c01_lr1e4_5000_20260623` | 23 | 0.6957 | 3500 | 1.0000 | 4500 | 0.0529 |
| `dvq_fixed_ratio03_h2048x3_c01_lr1e4_5000_20260623` | 69 | 0.3188 | 4000 | 0.3623 | 3000 | 0.0864 |
| `dvq_fixed_ratio03_h4096x4_c01_lr1e4_5000_20260623` | 69 | 0.3478 | 4500 | 0.3768 | 4500 | 0.0864 |

Reflection:

- The requested train-accuracy target is now achieved for the small split: Direct VQ-Plus reaches `0.6087` with `2048 x 3` and `0.6957` with `4096 x 4` on 23 training pairs.
- This is an overfit result, not generalization: eval accuracy remains near chance because the eval split contains held-out arithmetic pairs.
- Increasing capacity helps train overfit on `train_ratio=0.1`, but only modestly improves `train_ratio=0.3`.
- Explicit commit loss helps optimize continuous distance and train overfit, but does not solve held-out pair composition.
