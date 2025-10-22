# Transformer from Scratch

[![GitHub](https://img.shields.io/badge/GitHub-luojun132/transformer-blue)](https://github.com/luojun132/transformer)

从零实现的Transformer架构，包含完整的训练管道和消融实验系统。本项目完整实现了Transformer论文《Attention Is All You Need》中的核心组件，并在Tiny Shakespeare数据集上进行了系统的消融实验。

## 🎯 项目特点

- ✅ **从零实现**所有Transformer核心组件
- ✅ **完整的Encoder-Decoder架构**
- ✅ **系统的消融实验** (位置编码、注意力头数、层数)
- ✅ **自动化实验结果可视化**
- ✅ **支持语言建模和序列到序列任务**
- ✅ **完整的训练管道** (AdamW优化器、学习率调度、梯度裁剪)

## 📁 项目结构

```
transformer-from-scratch/
├── src/                          # 源代码
│   ├── model.py                  # Transformer模型实现
│   ├── trainer.py                # 训练器
│   ├── data_loader.py            # 数据加载与预处理
│   ├── utils.py                  # 工具函数
│   ├── experiment.py             # 实验管理器
│   └── seq2seq_data.py           # 序列到序列数据
├── configs/                      # 配置文件
│   ├── base.yaml                 # 基础训练配置
│   └── seq2seq.yaml              # 序列到序列配置
├── scripts/                      # 运行脚本
│   └── run.sh                    # 完整实验运行脚本
├── results/                      # 实验结果
│   ├── ablation/                 # 消融实验图表
│   │   ├── positional_encoding_results.png
│   │   ├── heads_results.png
│   │   ├── layers_results.png
│   │   └── ablation_report.md
│   └── training_curve.png        # 训练曲线
├── checkpoints/                  # 模型检查点目录
├── data/                         # 数据集
│   └── tiny_shakespeare.txt      # Tiny Shakespeare数据集
├── model_backups/                # 模型文件备份
├── requirements.txt              # Python依赖
├── README.md                     # 项目说明
├── run_ablation.py               # 消融实验主脚本
├── train.py                      # 单次训练脚本
└── train_seq2seq.py              # 序列到序列训练
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **内存**: 4GB+ RAM
- **磁盘空间**: 2GB+
- **推荐硬件**: GPU (CUDA) 用于加速训练

### 安装依赖

```bash
pip install -r requirements.txt
```

### 精确重现实验

使用随机种子42确保结果可重现：

```bash
# 方法1: 使用运行脚本
chmod +x scripts/run.sh
./scripts/run.sh

# 方法2: 直接运行消融实验
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_ablation.py
```

### 单次训练

```bash
python train.py
```

### 序列到序列训练

```bash
python train_seq2seq.py
```

## ⚙️ 硬件要求与训练时间

| 配置 | 硬件要求 | 训练时间 | 备注 |
|------|----------|----------|------|
| **消融实验** | CPU, 8GB RAM | ~2小时 | 完整的位置编码、头数、层数实验 |
| **单次训练** | CPU, 4GB RAM | ~30分钟 | 单个模型训练 |
| **GPU加速** | NVIDIA GPU, 4GB VRAM | ~20分钟 | 显著加速训练 |

## 📊 核心实现

### 模型架构

- **Multi-head Self-attention**: 缩放点积注意力机制
- **Position-wise Feed-Forward Networks**: 位置前馈网络
- **残差连接 + Layer Normalization**: 稳定训练的关键组件
- **正弦位置编码**: 提供序列位置信息
- **完整的Encoder-Decoder架构**: 支持序列到序列任务

### 训练特性

- **AdamW优化器**: 带权重衰减的Adam优化器
- **Cosine Annealing调度**: 余弦退火学习率调度
- **梯度裁剪**: 防止梯度爆炸
- **自动模型保存**: 智能检查点管理
- **训练过程可视化**: 实时损失曲线绘制

## 📈 实验结果

### 消融实验性能对比

| 实验配置 | 最终验证损失 | 最佳验证损失 | 关键发现 |
|----------|--------------|--------------|----------|
| **有位置编码** | 0.0987 | 0.0987 | 位置编码至关重要 |
| **无位置编码** | 2.8289 | 2.8289 | 性能下降28倍 |
| **2注意力头** | 0.1478 | 0.1478 | 基础性能表现 |
| **4注意力头** | 0.0987 | 0.0987 | 平衡的性能 |
| **8注意力头** | 0.0933 | 0.0933 | 多头注意力优势 |
| **2层Transformer** | 0.0987 | 0.0987 | 标准配置 |
| **4层Transformer** | 0.1232 | 0.1232 | 轻微过拟合 |
| **6层Transformer** | 0.0777 | 0.0777 | 最佳性能 |

### 关键发现

1. **位置编码的重要性**: 有位置编码的模型性能显著优于无位置编码模型
2. **多头注意力优势**: 增加注意力头数带来持续的性能提升
3. **适当模型深度**: 6层Transformer在数据集上表现最佳
4. **训练稳定性**: 所有配置都展现了良好的收敛性

## 🔬 实验详情

### 位置编码消融实验
验证位置编码在Transformer中的关键作用，比较有/无位置编码的性能差异。

### 注意力头数消融实验
探索多头注意力机制中头数对模型性能的影响，比较2头、4头、8头的表现。

### 层数消融实验
研究模型深度对性能的影响，比较2层、4层、6层Transformer的表现。

## 🛠️ 自定义配置

修改 `configs/base.yaml` 来调整实验设置：

```yaml
# 模型配置
d_model: 128
num_heads: 4
num_layers: 2
d_ff: 512
max_seq_len: 128
dropout: 0.1

# 训练配置
batch_size: 16
learning_rate: 0.002
weight_decay: 0.01
num_epochs: 8
random_seed: 42
```

## 📋 文件说明

### 核心文件
- `src/model.py`: Transformer模型完整实现
- `src/trainer.py`: 训练循环和模型保存
- `src/experiment.py`: 消融实验管理系统
- `run_ablation.py`: 消融实验主入口

### 结果文件
- `results/ablation/positional_encoding_results.png`: 位置编码实验图表
- `results/ablation/heads_results.png`: 注意力头数实验图表
- `results/ablation/layers_results.png`: 层数实验图表
- `results/ablation/ablation_report.md`: 完整实验分析报告

### 数据文件
- `data/tiny_shakespeare.txt`: 训练数据集
- `checkpoints/`: 模型检查点目录（文件较大没有上传）
- `model_backups/`: 大模型文件备份

- 使用 [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) 数据集进行验证


*项目最后更新: 2024年1月*  
*如遇问题，请提交Issue或联系维护者*
