# V3 code map

Scope: `../variance-versus-invariance`.

## Project entry

- Training entry: `../variance-versus-invariance/run_training.py`
  - Reads YAML config.
  - Builds `Trainer`.
  - Calls `prepare_data`, `build_model`, `train`.

- Evaluation entry: `../variance-versus-invariance/run_evaluation.py`
  - Uses `Tester`.
  - Supports `--pr_metrics`, `--vis_tsne`, `--confusion_mtx`, `--zero_shot_ood`, `--few_shot_ood`.

- Main trainer: `../variance-versus-invariance/trainer.py`
  - `prepare_data`: imports `dataloader.<name>`.
  - `build_model`: calls `model.factory.get_model`.
  - Training loop forwards model and passes tensors to `V3Loss`.

## Model

- Factory: `../variance-versus-invariance/model/factory.py`
  - `get_model(dataloader_name, model_config)` chooses the domain-specific Encoder/Decoder module.
  - For `phonenums`, lowercase letters, uppercase letters: uses `model/modules/phonenums.py`.

- Core autoencoder: `../variance-versus-invariance/model/autoencoder.py`
  - `CSAE`: line 7
  - `encode`: line 33
  - `quantize`: line 38
  - `forward`: line 48

`CSAE` structure:

```text
input sequence/fragments
  -> encoder
    -> emb_c: content embedding
    -> emb_s: style embedding
  -> VectorQuantize(emb_c)
    -> emb_c_vq, vq_indices, commit_loss
  -> decoder(emb_c_vq, emb_s)
```

The V5-relevant interface is:

```python
emb_c, emb_s = model.encode(x)
emb_c_vq, vq_indices, commit_loss = model.quantize(emb_c, freeze_codebook=True)
```

If V3 is used as a frozen pretrained encoder, likely freeze:

- `model.encoder`
- `model.vq`

and optionally skip or freeze:

- `model.decoder`

## V3 loss

File: `../variance-versus-invariance/model/v3_loss.py`

- `mpd`: line 8
- `V3Loss`: line 31
- pure loss `_compute_loss_pure`: line 45

Pure V3 statistics:

- `content_frag_var`: content varies across fragments inside a sample.
- `content_samp_var`: content vocabulary should be invariant across samples.
- `style_frag_var`: style should be invariant across fragments inside a sample.
- `style_samp_var`: style varies across samples.

Loss ratios:

- `content_loss = relu(r - content_frag_var / content_samp_var) / r`
- `style_loss = relu(r - style_samp_var / style_frag_var) / r`
- `sample_loss = relu(r - style_samp_var / content_samp_var) / r`
- `fragment_loss = relu(r - content_frag_var / style_frag_var) / r`

ELPIS already has a simplified copy of this in `../S3Plus/VQ/train.py`.

## PhoneNums / colored digit path

Likely closest to V5's "red 3 + green 5 -> 8" intuition.

- Config: `../variance-versus-invariance/cfg_v3_phonenums.yaml`
- Dataloader: `../variance-versus-invariance/dataloader/phonenums_dataloader.py`
  - `PhoneNumsDataset`: line 27
  - `__getitem__`: line 67
  - `get_dataloader`: line 114
- Model module: `../variance-versus-invariance/model/modules/phonenums.py`
  - Used for phonenums and letter image domains.

PhoneNums data behavior:

- Each sample is an image sequence, e.g. a phone number rendered in one color/style.
- The dataloader cuts `n_fragments` digit crops of width `fragment_len`.
- Content labels are digits.
- Style label is color.
- Returned batch shape is `[batch, n_fragments, C, H, W]`.

This sequence/fragments format maps well to ELPIS triplets if we set `n_fragments=3` or adapt the triplet dataloader to emit `[batch, 3, C, H, W]`.

## Available pretrained checkpoints

Local checkpoints found:

```text
../variance-versus-invariance/logs/phonenums_trial/cp_epoch747.pt
../variance-versus-invariance/logs/phonenums_trial/cp_epoch751.pt
../variance-versus-invariance/logs/phonenums_trial/cp_epoch753.pt
../variance-versus-invariance/logs/phonenums_trial/cp_epoch9999.pt
../variance-versus-invariance/logs/lowercase_letters_run1/cp_epoch*.pt
../variance-versus-invariance/logs/uppercase_letters_run*/cp_epoch*.pt
```

For V5 colored digits, start with:

```text
../variance-versus-invariance/logs/phonenums_trial/config.yaml
../variance-versus-invariance/logs/phonenums_trial/cp_epoch9999.pt
```

Checkpoint loading pattern in V3:

```python
save_info = torch.load(cp_path)
model.load_state_dict(save_info["model"])
```

## Shape notes for V5

V3 image modules expect:

```text
[batch_size, n_segments, 3, n_feature, fragment_len]
```

ELPIS currently stacks triplets as:

```text
[3 * batch_size, 3, H, W]
```

So V5 needs a small shape adapter:

```python
triplet = torch.stack([a, b, c], dim=1)  # [B, 3, C, H, W]
emb_c, emb_s = v3_model.encode(triplet)
```

