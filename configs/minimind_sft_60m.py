# Full-parameter supervised fine-tuning of our 62M vanilla Transformer.
# Architecture must match the pretraining checkpoint exactly.
model = dict(
    vocab_size=6400,
    block_size=768,
    n_layer=8,
    n_head=8,
    d_model=768,
    d_ff=3072,
    dropout=0.1,
    attention_type="mha",
    position_embedding_type="learned_absolute",
)

train = dict(
    data_dir="data/minimind/sft_mini",
    tokenizer="minimind",
    init_from="checkpoints/minimind_pretrain_full_60m/best.pt",
    out_dir="checkpoints/minimind_sft_60m",
    epochs=2,
    # Per-GPU batch. With 2 GPUs this is 256 conversations per optimizer step.
    batch_size=128,
    gradient_accumulation_steps=1,
    eval_interval=500,
    eval_batches=30,
    log_interval=10,
    learning_rate=1e-5,
    min_lr=1e-6,
    warmup_steps=100,
    weight_decay=0.1,
    grad_clip=1.0,
    dtype="bfloat16",
    compile=True,
    fused_optimizer=True,
    seed=1337,
    wandb_mode="online",
    wandb_entity="Zjunlp-Xubqpanda",
    wandb_project="Memory-Model",
    wandb_run_name="minimind-sft-60m-vanilla",
    wandb_tags=["baseline", "minimind", "sft", "assistant-only-loss", "60m"],
    wandb_init_timeout=30,
    wandb_fallback_offline=True,
)
