model = dict(
    vocab_size=256,
    block_size=64,
    n_layer=2,
    n_head=4,
    d_model=128,
    d_ff=512,
    dropout=0.0,
)

train = dict(
    data_dir="data/toy",
    tokenizer="byte",
    out_dir="checkpoints/tiny_debug",
    batch_size=16,
    gradient_accumulation_steps=1,
    max_steps=200,
    eval_interval=50,
    eval_batches=10,
    log_interval=10,
    learning_rate=3e-4,
    min_lr=3e-5,
    warmup_steps=20,
    weight_decay=0.1,
    grad_clip=1.0,
    dtype="float32",
    compile=False,
    seed=1337,
)
