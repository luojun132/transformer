import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from typing import Dict, List, Tuple, Optional


def set_seed(seed: int = 42):
    """设置随机种子以确保结果可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_detailed(model: torch.nn.Module) -> Dict:
    """详细统计模型各模块参数数量"""
    details = {}
    total_params = 0

    for name, module in model.named_children():
        module_params = count_parameters(module)
        details[name] = module_params
        total_params += module_params

    details['total'] = total_params
    return details


def print_model_summary(model: torch.nn.Module):
    """打印模型摘要信息"""
    print("=" * 80)
    print("Model Summary")
    print("=" * 80)

    param_details = count_parameters_detailed(model)

    for name, params in param_details.items():
        if name != 'total':
            print(f"{name:20} {params:>10,} parameters")

    print("-" * 80)
    print(f"{'TOTAL':20} {param_details['total']:>10,} parameters")
    print("=" * 80)


def save_training_history(train_losses: List[float], val_losses: List[float],
                          filepath: str = 'results/training_history.json'):
    """保存训练历史到JSON文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': len(train_losses),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)


def load_training_history(filepath: str = 'results/training_history.json') -> Dict:
    """从JSON文件加载训练历史"""
    with open(filepath, 'r') as f:
        history = json.load(f)
    return history


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                         save_path: Optional[str] = None):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制对数尺度损失曲线
    plt.subplot(1, 2, 2)
    plt.semilogy(train_losses, label='Training Loss', linewidth=2)
    plt.semilogy(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    plt.show()


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """创建padding mask"""
    # seq: [batch_size, seq_len]
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
    return mask


def create_look_ahead_mask(seq_len: int) -> torch.Tensor:
    """创建look-ahead mask用于decoder"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask  # [seq_len, seq_len]


def calculate_perplexity(loss: float) -> float:
    """根据损失计算困惑度"""
    return np.exp(loss)


def model_size_in_mb(model: torch.nn.Module) -> float:
    """计算模型大小（MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


def get_device() -> torch.device:
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def print_device_info():
    """打印设备信息"""
    device = get_device()
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

    elif device.type == 'mps':
        print("Using Apple Silicon GPU (MPS)")


class Timer:
    """计时器类"""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        print(f"{self.name} took {elapsed_time:.2f} seconds")


def generate_text(model: torch.nn.Module, tokenizer, start_text: str,
                  max_length: int = 50, temperature: float = 1.0) -> str:
    """使用训练好的模型生成文本"""
    model.eval()
    device = get_device()

    # 编码起始文本
    encoded = tokenizer.encode(start_text)
    input_ids = torch.tensor(encoded.ids).unsqueeze(0).to(device)

    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(generated)
            next_token_logits = logits[0, -1, :] / temperature

            # 应用softmax获取概率
            probs = torch.softmax(next_token_logits, dim=-1)

            # 从分布中采样
            next_token = torch.multinomial(probs, num_samples=1)

            # 添加到生成序列
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            # 如果生成了结束标记则停止
            if next_token.item() == tokenizer.token_to_id("[EOS]"):
                break

    # 解码生成的文本
    generated_text = tokenizer.decode(generated[0].cpu().numpy())
    return generated_text


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """计算模型梯度的L2范数"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_parameter_norm(model: torch.nn.Module) -> float:
    """计算模型参数的L2范数"""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class EarlyStopping:
    """早停类"""

    def __init__(self, patience: int = 7, min_delta: float = 0, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best:
                self.best_state_dict = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_state_dict = model.state_dict().copy()

    def restore_best_weights(self, model: torch.nn.Module):
        """恢复最佳权重"""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


def analyze_attention_patterns(attention_weights: torch.Tensor,
                               tokens: List[str],
                               layer_idx: int = 0,
                               head_idx: int = 0):
    """分析注意力模式"""
    attn = attention_weights[layer_idx][head_idx].cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')

    # 设置刻度标签
    if len(tokens) <= 20:  # 只在token数量较少时显示标签
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.yticks(range(len(tokens)), tokens)

    plt.tight_layout()
    plt.show()

    return attn


def save_config(config: Dict, filepath: str = 'results/config.json'):
    """保存配置到JSON文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    config_with_timestamp = config.copy()
    config_with_timestamp['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

    with open(filepath, 'w') as f:
        json.dump(config_with_timestamp, f, indent=2)


def load_config(filepath: str = 'results/config.json') -> Dict:
    """从JSON文件加载配置"""
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:.0f}m {seconds:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"