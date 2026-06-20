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

## Commands

From an allocated GPU shell:

```bash
cd /home/xuanjie.liu/Projects/v5
python -m src.train --config configs/uppercase_symm.yaml
python -m src.train --config configs/uppercase_nosymm.yaml
```

Short GPU smoke:

```bash
cd /home/xuanjie.liu/Projects/v5
python -m src.train --config configs/uppercase_symm.yaml --run_name smoke_symm --max_steps 5 --eval_interval 5 --batch_size 4
python -m src.train --config configs/uppercase_nosymm.yaml --run_name smoke_nosymm --max_steps 5 --eval_interval 5 --batch_size 4
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
- `best_1.pt`, `best_2.pt`, `best_3.pt`
- `vis/metrics.png`
- `vis/train_addition_heatmap.png`
- `vis/eval_addition_heatmap.png`
- `vis/eval_confusion.png`
- `vis/triplet_grid_step_<N>.png`

## Notes

The V3 codebook atom id is not assumed to match the letter id. The run estimates an atom-to-letter majority mapping from uppercase train pages and saves it in `v3_codebook_mapping.json`.

