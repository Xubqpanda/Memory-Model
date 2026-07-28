# Full-data counterpart of minimind_pretrain_60m.py.
# It keeps exactly the same 62M vanilla Transformer architecture so the
# difference from the mini run isolates the effect of pretraining data.
model = dict(
    vocab_size=6400,
    block_size=768,
    n_layer=8,
    n_head=8,
    d_model=768,
    d_ff=3072,
    dropout=0.1,
)

train = dict(
    data_dir="data/minimind/pretrain_full",
    tokenizer="minimind",
    out_dir="checkpoints/minimind_pretrain_full_60m",
    # Per-GPU batch. The trainer preserves 196,608 global tokens/update:
    # 1 GPU -> accumulation 2; 2 GPUs -> accumulation 1.
    batch_size=128,
    gradient_accumulation_steps=2,
    target_tokens_per_step=196608,
    # 11,123 × 196,608 = 2,186,870,784 sampled tokens, almost exactly one
    # pass over the 2,186,853,770-token training split.
    max_steps=11123,
    eval_interval=250,
    eval_batches=30,
    log_interval=10,
    learning_rate=4e-4,
    min_lr=4e-5,
    warmup_steps=500,
    weight_decay=0.1,
    grad_clip=1.0,
    dtype="bfloat16",
    compile=True,
    fused_optimizer=True,
    seed=1337,
    wandb_mode="online",
    wandb_entity="Zjunlp-Xubqpanda",
    wandb_project="Memory-Model",
    wandb_run_name="minimind-pretrain-full-60m-vanilla",
    wandb_tags=["baseline", "minimind", "pretrain", "full-data", "60m", "vanilla-transformer"],
    wandb_init_timeout=30,
    wandb_fallback_offline=True,
)
