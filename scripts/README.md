# Script organization

命令行入口按生命周期分组：

```text
scripts/
├── data/          数据下载、清洗、tokenize 和二进制化
├── train/         预训练、微调和后续 memory 实验训练
├── inference/     文本生成和网页推理
├── eval/          固定评测与 checkpoint 对比
└── memory/        memory 轨迹生成和独立机制实验
```

命名约定：

- 文件名使用小写 `snake_case`。
- 使用动词描述操作，例如 `prepare_tinystories.py` 和 `generate.py`。
- 训练入口用训练阶段命名，例如 `pretrain.py` 和 `sft.py`。
- 可复用的模型、数据与生成逻辑放在 `src/`，`scripts/` 只负责解析参数和组织运行流程。
- 相对路径统一相对于项目根目录解析，避免因当前工作目录不同而改变行为。

当前训练阶段入口：

```text
scripts/data/prepare_minimind_sft.py  MiniMind 对话数据 ChatML 编码与 loss mask
scripts/train/pretrain.py             next-token 预训练
scripts/train/sft.py                  assistant-only loss 全参数监督微调
scripts/train/dpo.py                  chosen/rejected 直接偏好优化
scripts/inference/web_chat.py         SFT ChatML 多轮对话网页
scripts/eval/compare_checkpoints.py   固定评测集上的 checkpoint 对比
scripts/memory/smoke_metis_lite.py    Metis-lite 动态读写 smoke test
```
