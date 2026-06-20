# V5 working notes

V5 的目标：把 ELPIS 的 triplet addition data + latent symmetry constraints，接到 V3 pretrained content/style encoder 上，让模型在混合 style 的 dynamics 里仍能靠 content-level symmetry 泛化。例如只见过红色 3、绿色 5，也能在 content 语义上推出 8。

当前目录主要放 V5 相关论文、代码地图和整合草图。真正的源代码仍在两个上游项目里：

- ELPIS / S3Plus: `../S3Plus/VQ`
- V3: `../variance-versus-invariance`
- Papers: `papers/ELPIS.pdf`, `papers/v3.pdf`

阅读 ELPIS 时按需求忽略 `2.2`, `2.3`, `4.8`。正文蓝点实验对应 setting：

```text
../S3Plus/VQ/exp/2025.05.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Fullsymm/train_config.py
```

快速入口：

- `ELPIS_CODEMAP.md`: ELPIS 训练入口、dataset、model、loss、eval 的代码地图。
- `V3_CODEMAP.md`: V3 encoder/decoder、V3Loss、dataloader、checkpoint 的代码地图。
- `V5_INTEGRATION_SKETCH.md`: V5 初步拼接方案、可复用代码、待决策点。

已确认的 baseline 信号：

- ELPIS 指定 Fullsymm setting 在 `PIPELINE_EVAL/all_results_summary.json` 中，20 个 sub-exp 的 `eval_set_one2n_accu` 和 `eval_set_one2one_accu` 都是 `1.0 +/- 0.0`。
- 单个 sub-exp `1/plus_eval.txt` 后期记录中，train/eval plus accuracy、embedding consistency、matching rate、NNA score 都达到 `1.0`。

