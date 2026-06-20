# ELPIS code map

Scope: `../S3Plus/VQ` plus shared helpers in `../S3Plus`.

## Experiment entry

- Batch entry: `../S3Plus/VQ/batch_train.py`
  - Usage: `python batch_train.py <exp_name> [<exp_name> ...]`
  - Adds `../S3Plus/VQ/exp` to `sys.path`, imports each experiment's `train_config.py`, then runs `PlusTrainer`.
  - Creates sub-experiments `1..num_sub_exp` under the experiment folder.

- Main trainer: `../S3Plus/VQ/train.py`
  - `PlusTrainer`: line 94
  - `one_epoch`: line 165
  - `plus_loss`: line 356
  - `associative_loss`: line 424
  - `operation_loss_z`: line 448
  - embedded V3-style loss helper `cal_v3_loss`: line 518

## Highlighted setting

File:

```text
../S3Plus/VQ/exp/2025.05.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Fullsymm/train_config.py
```

Important config values:

- Data:
  - train: `../S3Plus/dataset/single_style_pairs(0,20)_tripleSet/train`
  - eval: `test_1`, `test_2`
  - single image eval: `../S3Plus/dataset/(0,20)-FixedPos-oneStyle`
- Latent/content:
  - `latent_embedding_1 = 2`
  - `latent_embedding_2 = 0`
  - `embedding_dim = 1`
  - `embeddings_num = 10`
  - content code size is `latent_embedding_1 * embedding_dim = 2`
- Addition module:
  - `plus_by_embedding = True`
  - `plus_by_zcode = False`
  - `network_config.plus.plus_unit = 1024`
  - `network_config.plus.n_hidden_layers = 2`
- Loss weights:
  - `z_plus_loss_scalar = 0.02`
  - `associative_z_loss_scalar = 0.02`
  - `plus_recon_loss_scalar = 3`
  - `VQPlus_eqLoss_scalar = 0.5`
  - `commutative_z_loss_scalar = 0`
- Symmetry:
  - `is_full_symm = True`
  - `is_assoc = False`
  - `is_zc_based_assoc = True`
  - `is_assoc_within_batch = True`

## Dataset format

- Triplet dataset class: `../S3Plus/dataloader_plus.py`
  - `MultiImgDataset` reads each sample as a subdirectory.
  - Inside each subdirectory, sorted image files become the three tensors `(a, b, c)`.
  - Example:

```text
../S3Plus/dataset/single_style_pairs(0,20)_tripleSet/train/1-15-o-blue/
  a-1.png
  b-15.png
  c-16.png
```

- Single image dataset class: `../S3Plus/dataloader.py`
  - `SingleImgDataset` reads one image per sample.
  - `load_enc_eval_data_with_style` expects names like `num-shape-color.png`.

- Dataloader factory: `../S3Plus/utils.py`
  - `init_dataloaders(config)` supports predefined train/eval sets and optional random split.
  - `plus_eval_set_path` can be a list.

## Model

File: `../S3Plus/VQ/VQVAE.py`

- `VQVAE`: line 256
- `plus`: line 297
  - Concats two content vectors, sends through `plus_net`, then VQ-quantizes.
- `batch_encode_to_z`: line 303
  - Encoder output is split into content `z_c` and style `z_s`.
  - If `isVQStyle=False`, only content is VQ-quantized; style stays continuous.
- `find_indices`: line 330
  - Converts quantized content embeddings to decimal code indices for eval.

For the highlighted setting, `latent_code_2=0`, so ELPIS is effectively content-only.

## Loss wiring

High-level `one_epoch` flow:

1. Stack triplet images into `[3 * batch, C, H, W]`.
2. Encode all images: `e_all, e_q_loss, z_all = model.batch_encode_to_z(data_all)`.
3. Reconstruct all images with VQ content.
4. Compute VAE reconstruction loss.
5. Compute supervised plus loss on `(a,b)->c`.
6. Compute operation loss, especially symmetry/associativity variants.
7. Optionally compute V3-style variance-invariance losses.

`plus_loss(za, zb, zc, imgs_c)`:

- Chooses style from `a` or `b` unless disabled.
- Uses only content part for `model.plus`.
- Decodes predicted content plus selected style.
- Loss pieces:
  - reconstruct predicted `c` image: `plus_recon_loss_scalar`
  - match predicted latent to encoded `c`: `z_plus_loss_scalar`
  - quantization loss on plus result: `VQPlus_eqLoss_scalar`

`associative_loss(z_a, z_b, z_c)`:

- Computes multiple paths:
  - `h(h(a,b),c)`
  - `h(h(a,c),b)`
  - `h(b,h(a,c))`
  - `h(a,h(b,c))`
- With `is_full_symm=True`, enforces:
  - `h(h(a,b),c) == h(h(a,c),b)`
  - `h(a,h(b,c)) == h(b,h(a,c))`
- With `is_assoc=True`, also enforces pure associativity:
  - `h(h(a,b),c) == h(a,h(b,c))`

This is the ELPIS physical symmetry hook most relevant to V5.

## Evaluation hooks

- Pipeline eval: `../S3Plus/VQ/eval_pipeline.py`
- Plus eval helpers: `../S3Plus/VQ/eval_plus_nd.py`
- Single image / orderliness eval:
  - `../S3Plus/VQ/eval_multi_style.py`
  - `../S3Plus/VQ/two_dim_num_vis.py`
  - `../S3Plus/VQ/plot_multistyle_zc.py`

Metrics to carry into V5:

- `one2n_accu`, `one2one_accu`, and cycle variants for addition.
- `emb_self_consistency`, `emb_label_consistency`.
- `one2n_match_rate`, `one2one_matching_rate` for content-code label alignment.
- `nna_score` / orderliness for number-line structure.

## Existing V3 trace inside ELPIS

`../S3Plus/VQ/train.py` already has:

- config flags: `use_v3_loss`, `v3_relativity`
- `cal_v3_loss` and `_compute_loss_pure`, copied/adapted from V3.

This means V5 can either:

- reuse this lightweight ELPIS-local V3 loss for triplet-shaped batches, or
- import the original V3 `V3Loss` if preserving exact repo behavior matters.

