# RoPE ablation counterpart of minimind_pretrain_60m.py.
# It changes only the position method so results can be compared fairly with
# the learned-absolute-position baseline.
model = dict(
    vocab_size=6400,
    block_size=768,
    n_layer=8,
    n_head=8,
    d_model=768,
    d_ff=3072,
    dropout=0.1,
    attention_type="mha",
    position_embedding_type="rope",
    rope_theta=10_000.0,
)

train = dict(
    data_dir="data/minimind/pretrain_mini",
    tokenizer="minimind",
    out_dir="checkpoints/minimind_pretrain_rope_60m",
    batch_size=128,
    gradient_accumulation_steps=2,
    target_tokens_per_step=196608,
    max_steps=6400,
    eval_interval=200,
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
    wandb_run_name="minimind-pretrain-mini-60m-rope",
    wandb_tags=["ablation", "minimind", "pretrain", "60m", "mha", "rope"],
    wandb_init_timeout=30,
    wandb_fallback_offline=True,
)
