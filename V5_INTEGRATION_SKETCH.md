# V5 integration sketch

## Research intent

Use V3 pretrained content/style disentanglement to make perceptual input already style-robust, then apply ELPIS content-level addition and symmetry constraints. The hope is:

```text
mixed-style perceptual triplet
  -> V3 content encoder
  -> content symbols / content embeddings
  -> ELPIS plus_net + full symmetry
  -> generalizable addition dynamics
```

Example target behavior:

```text
red 3 + green 5 -> content 8
```

The output can initially be evaluated in content/code space, before requiring a specific output style image.

## Minimal V5 architecture

First prototype:

1. Load V3 pretrained `CSAE`.
2. Freeze V3 encoder and content VQ codebook.
3. Read ELPIS-style triplet folders, but allow mixed styles in `a`, `b`, `c`.
4. Convert triplet batch to V3 shape `[B, 3, C, H, W]`.
5. Encode with V3:

```python
emb_c, emb_s = v3.encode(triplet)
emb_c_vq, vq_indices, commit_loss = v3.quantize(emb_c, freeze_codebook=True)
```

6. Train an ELPIS-style `plus_net` on content only:

```text
plus_net(emb_c_vq[:, 0], emb_c_vq[:, 1]) -> predicted content for c
```

7. Apply ELPIS losses:

- supervised content match: predicted `c` vs V3 content of target `c`
- full symmetry: same multi-path consistency as `PlusTrainer.associative_loss`
- optional reconstruction: decode predicted content with a chosen style embedding using V3 decoder

## Loss choices

### Content-only first

Recommended first experiment:

- Freeze V3 encoder/VQ.
- Train only `plus_net`.
- Use content embedding loss:

```text
MSE(pred_c_content_vq, target_c_content_vq)
```

- Use ELPIS full symmetry on content embeddings.
- Evaluate hard matching by nearest V3 content code / digit label.

This avoids decoder/style questions and directly tests whether symmetry over V3 content supports addition generalization.

### Add reconstruction later

Once content accuracy works:

- Decode predicted `c` using:
  - `c` target style for teacher-forced reconstruction, or
  - `a`/`b` style for controlled style transfer, or
  - a sampled style to test style-independent content.

Possible reconstruction:

```python
x_hat_c = v3.decode(pred_c_content_vq, emb_s[:, 2])
```

## Copy/reuse candidates

From ELPIS:

- `PlusTrainer.plus_loss`: supervision shape and content/style split idea.
- `PlusTrainer.associative_loss`: full symmetry implementation.
- `PlusTrainer.zc_based_associative_z`: in-batch latent tuple sampling.
- `VQVAE.plus`: MLP + VQ output pattern, though V5 may use V3 VQ instead.
- `eval_plus_nd.py`: plus accuracy logic, but adapt code-index source to V3 `vq_indices`.

From V3:

- `model.autoencoder.CSAE`: pretrained encoder/VQ/decoder wrapper.
- `model.factory.get_model`: build correct model from YAML.
- `model.v3_loss.V3Loss` or ELPIS-local `cal_v3_loss`.
- `dataloader/phonenums_dataloader.py`: colored digit image assumptions and fragment shape.

## Data options

### Option A: ELPIS folder format, mixed style files

Keep the ELPIS triplet folder convention:

```text
train/3-5-mixed/
  a-3-red.png
  b-5-green.png
  c-8-blue.png
```

Pros:

- Minimal changes to ELPIS sampling/eval.
- Easy to create seen/unseen addend splits.

Cons:

- Need to make sure image size/normalization matches V3 phonenums encoder.

### Option B: V3 sequence format, triplets as fragments

Represent each addition problem as one sample with 3 fragments:

```text
[x_a, x_b, x_c]
```

Pros:

- Natural fit for V3 encoder and V3 loss.

Cons:

- Need new labels and plus-eval logic.

## Evaluation plan

Core matrix:

- seen addend pair + seen style mix
- unseen addend pair + seen style mix
- seen addend pair + unseen style mix
- unseen addend pair + unseen/mixed style

Metrics:

- Hard content-code matching: predicted content code equals target digit's V3 code.
- Digit-label matching: map V3 code to digit by majority vote, then check `a+b=c`.
- Cycle/self consistency from ELPIS plus eval.
- Optional reconstruction MSE/visual grid after decoder is included.

Useful baseline comparisons:

- ELPIS Fullsymm single-style baseline: currently `1.0 +/- 0.0`.
- ELPIS no-symmetry on same V5 data.
- V3 frozen encoder + plus supervised but no symmetry.
- V3 frozen encoder + plus + full symmetry.

## Early implementation shape

Potential new files under `v5/src` later:

```text
v5/src/v3_loader.py          # build/load pretrained V3 model from config+checkpoint
v5/src/triplet_dataset.py    # mixed-style triplet dataset, returns [B,3,C,H,W]
v5/src/plus_model.py         # ELPIS-style content plus_net
v5/src/train_v5.py           # frozen V3 encoder + train plus/symmetry
v5/src/eval_v5.py            # content-code plus eval
v5/configs/*.yaml            # paths and loss weights
```

## Immediate open decisions

- Which V3 checkpoint is canonical for colored digits: likely `phonenums_trial/cp_epoch9999.pt`.
- Whether V3 content embedding dimension `d_emb_c=512` should feed plus_net directly, or whether to train a small projection before plus.
- Whether `plus_net` output should be quantized by V3's existing VQ codebook after prediction.
- Whether V3 encoder is fully frozen in the first V5 experiment.
- How to generate mixed-style triplets with the same image preprocessing as V3.

## First runnable prototype target

The smallest useful prototype should:

1. Load `phonenums_trial` V3 checkpoint.
2. Take triplets of colored digits.
3. Encode all three images through frozen V3.
4. Train `plus_net` on V3 content VQ embeddings.
5. Add ELPIS `is_full_symm` style loss.
6. Report unseen-pair content-code accuracy.

