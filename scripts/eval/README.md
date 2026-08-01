# Evaluation scripts

Run the fixed held-out chat comparison:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/eval/compare_checkpoints.py \
  --checkpoint sft=checkpoints/minimind_sft_60m/best.pt \
  --checkpoint dpo=checkpoints/minimind_dpo_60m/best.pt \
  --greedy
```

Use the same prompt, ChatML template, generation budget, decoding parameters,
and seed for every checkpoint. Results are written under `evals/results/` and
are ignored by Git.
