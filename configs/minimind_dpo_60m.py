# Direct Preference Optimization starting from the best SFT checkpoint.
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
    data_dir="data/minimind/dpo",
    tokenizer="minimind",
    init_from="checkpoints/minimind_sft_60m/best.pt",
    reference_from="checkpoints/minimind_sft_60m/best.pt",
    out_dir="checkpoints/minimind_dpo_60m",
    epochs=1,
    batch_size=32,
    gradient_accumulation_steps=1,
    beta=0.1,
    label_smoothing=0.0,
    average_log_probs=False,
    eval_interval=100,
    eval_batches=20,
    log_interval=10,
    learning_rate=5e-7,
    min_lr=5e-8,
    warmup_steps=20,
    weight_decay=0.0,
    grad_clip=1.0,
    dtype="bfloat16",
    # Keep policy/reference numerical kernels aligned. The dataset is small
    # enough that eager execution is already fast, and compilation can create
    # tiny initial log-probability differences against an uncompiled reference.
    compile=False,
    fused_optimizer=True,
    seed=1337,
    wandb_mode="online",
    wandb_entity="Zjunlp-Xubqpanda",
    wandb_project="Memory-Model",
    wandb_run_name="minimind-dpo-60m-vanilla",
    wandb_tags=["baseline", "minimind", "dpo", "preference", "60m"],
    wandb_init_timeout=30,
    wandb_fallback_offline=True,
)
