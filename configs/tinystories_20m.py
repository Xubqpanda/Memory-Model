# About 20M parameters with GPT-2's 50,257-token vocabulary.
model = dict(
    vocab_size=50257,
    block_size=256,
    n_layer=8,
    n_head=8,
    d_model=256,
    d_ff=1024,
    dropout=0.1,
    attention_type="mha",
    position_embedding_type="learned_absolute",
)

train = dict(
    data_dir="data/tinystories_gpt2",
    tokenizer="gpt2",
    out_dir="checkpoints/tinystories_20m",
    # Per-GPU batch. With target_tokens_per_step, accumulation resolves to:
    # 1 GPU -> 2 accumulation steps; 2 GPUs -> 1 accumulation step.
    batch_size=128,
    gradient_accumulation_steps=2,
    target_tokens_per_step=65536,
    # 8,000 × 65,536 ≈ 524M sampled tokens, close to one full TinyStories pass.
    max_steps=8000,
    eval_interval=250,
    eval_batches=50,
    log_interval=10,
    learning_rate=6e-4,
    min_lr=6e-5,
    warmup_steps=500,
    weight_decay=0.1,
    grad_clip=1.0,
    dtype="bfloat16",
    compile=True,
    seed=1337,
    wandb_mode="online",
    wandb_entity="Zjunlp-Xubqpanda",
    wandb_project="Memory-Model",
    wandb_run_name="tinystories-20m-baseline",
    wandb_tags=["baseline", "tinystories", "20m"],
    wandb_init_timeout=30,
    wandb_fallback_offline=True,
)
