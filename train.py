import torch
import yaml
from src.model import TransformerLM
from src.trainer import TransformerTrainer
import os


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # 加载配置
    config = load_config('configs/base.yaml')

    # 设置随机种子
    torch.manual_seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['random_seed'])

    # 手动加载数据
    print("正在手动加载数据集...")

    # 导入手动数据加载器
    try:
        from download_data import load_manual_data
        train_loader, val_loader, tokenizer = load_manual_data(
            batch_size=config['batch_size'],
            max_length=config['max_seq_len']
        )
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 初始化模型
    print("初始化模型...")
    model = TransformerLM(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )

    # 打印模型统计信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")

    # 初始化训练器
    trainer = TransformerTrainer(model, train_loader, val_loader, config)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()