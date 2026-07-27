# Script organization

命令行入口按生命周期分组：

```text
scripts/
├── data/          数据下载、清洗、tokenize 和二进制化
├── train/         预训练、微调和后续 memory 实验训练
└── inference/     文本生成、推理和后续评测入口
```

命名约定：

- 文件名使用小写 `snake_case`。
- 使用动词描述操作，例如 `prepare_tinystories.py` 和 `generate.py`。
- 训练入口用训练阶段命名，例如 `pretrain.py`；未来可加入 `finetune.py`。
- 可复用的模型、数据与生成逻辑放在 `src/`，`scripts/` 只负责解析参数和组织运行流程。
- 相对路径统一相对于项目根目录解析，避免因当前工作目录不同而改变行为。
