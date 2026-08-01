# Fixed evaluation data

`fixed_chat_eval.jsonl` is a manually curated, permanently held-out behavioral
suite for comparing Memory-Model checkpoints. It must never be included in
pretraining, SFT, DPO, RL, rejection-sampling, or prompt-generation data.

Each row contains:

- `id`: stable unique identifier;
- `category`: evaluation slice;
- `messages`: ChatML-compatible conversation ending in a user message;
- `checks`: deterministic checks when the task has an objective answer;
- `max_new_tokens`: per-sample generation budget.

Open-ended rows intentionally have an empty `checks` list. They are still used
for EOS, repetition, length, role-leakage, and human pairwise evaluation.
