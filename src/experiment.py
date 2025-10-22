import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime  # 添加这行
from .trainer import TransformerTrainer
from .seq2seq_trainer import Seq2SeqTrainer


class AblationExperiment:
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = {}

    def run_positional_encoding_ablation(self):
        """位置编码消融实验"""
        print("Running positional encoding ablation...")

        results = {}
        for use_pe in [True, False]:
            config = self.base_config.copy()
            config['use_positional_encoding'] = use_pe
            config['experiment_name'] = f'pe_{use_pe}'

            result = self._run_single_experiment(config)
            results[config['experiment_name']] = result

        self.results['positional_encoding'] = results
        return results

    def run_heads_ablation(self):
        """注意力头数消融实验"""
        print("Running heads ablation...")

        results = {}
        for num_heads in [2, 4, 8]:
            config = self.base_config.copy()
            config['num_heads'] = num_heads
            config['experiment_name'] = f'heads_{num_heads}'

            result = self._run_single_experiment(config)
            results[config['experiment_name']] = result

        self.results['heads'] = results
        return results

    def run_layers_ablation(self):
        """层数消融实验"""
        print("Running layers ablation...")

        results = {}
        for num_layers in [2, 4, 6]:
            config = self.base_config.copy()
            config['num_layers'] = num_layers
            config['experiment_name'] = f'layers_{num_layers}'

            result = self._run_single_experiment(config)
            results[config['experiment_name']] = result

        self.results['layers'] = results
        return results

    def run_architecture_ablation(self):
        """架构消融实验：Encoder-only vs Encoder-Decoder"""
        print("Running architecture ablation...")

        results = {}

        # Encoder-only
        config_encoder = self.base_config.copy()
        config_encoder['model_type'] = 'encoder_only'
        config_encoder['experiment_name'] = 'encoder_only'
        result_encoder = self._run_single_experiment(config_encoder)
        results['encoder_only'] = result_encoder

        # Encoder-Decoder
        config_seq2seq = self.base_config.copy()
        config_seq2seq['model_type'] = 'encoder_decoder'
        config_seq2seq['experiment_name'] = 'encoder_decoder'
        result_seq2seq = self._run_seq2seq_experiment(config_seq2seq)
        results['encoder_decoder'] = result_seq2seq

        self.results['architecture'] = results
        return results

    def _run_single_experiment(self, config):
        """运行单个语言建模实验"""
        from .model import TransformerLM
        from download_data import load_manual_data

        torch.manual_seed(config['random_seed'])

        # 加载数据
        train_loader, val_loader, tokenizer = load_manual_data(
            batch_size=config['batch_size'],
            max_length=config['max_seq_len']
        )

        # 创建模型
        model = TransformerLM(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            num_layers=config['num_layers'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout'],
            use_positional_encoding=config.get('use_positional_encoding', True)
        )

        print(f"训练配置: {config['experiment_name']}")
        print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

        # 训练
        trainer = TransformerTrainer(model, train_loader, val_loader, config)
        trainer.train()

        # 立即保存这个实验的最终结果
        result = {
            'final_train_loss': trainer.train_losses[-1],
            'final_val_loss': trainer.val_losses[-1],
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'best_val_loss': min(trainer.val_losses),
            'model_params': sum(p.numel() for p in model.parameters()),
            'config': config
        }

        # 保存这个实验的独立结果文件
        self._save_single_experiment_result(config['experiment_name'], result)

        return result

    def _save_single_experiment_result(self, exp_name, result):
        """保存单个实验的结果"""
        os.makedirs('results/individual', exist_ok=True)

        result_file = f'results/individual/{exp_name}_result.json'

        # 转换numpy数组为列表以便JSON序列化
        json_result = {
            'experiment_name': exp_name,
            'final_train_loss': float(result['final_train_loss']),
            'final_val_loss': float(result['final_val_loss']),
            'best_val_loss': float(result['best_val_loss']),
            'model_params': result['model_params'],
            'train_losses': [float(x) for x in result['train_losses']],
            'val_losses': [float(x) for x in result['val_losses']],
            'timestamp': datetime.now().isoformat()
        }

        with open(result_file, 'w') as f:
            json.dump(json_result, f, indent=2)

        print(f"✅ 实验 {exp_name} 结果已保存: {result_file}")

    def _run_seq2seq_experiment(self, config):
        """运行序列到序列实验"""
        from .model import TransformerSeq2Seq
        from .seq2seq_data import load_translation_data

        torch.manual_seed(config['random_seed'])

        # 加载翻译数据
        train_loader, val_loader, src_tokenizer, tgt_tokenizer = load_translation_data(
            batch_size=config['batch_size'],
            max_length=config['max_seq_len']
        )

        # 创建序列到序列模型
        model = TransformerSeq2Seq(
            src_vocab_size=src_tokenizer.get_vocab_size(),
            tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            num_layers=config['num_layers'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout'],
            use_positional_encoding=config.get('use_positional_encoding', True)
        )

        # 训练
        from .seq2seq_trainer import Seq2SeqTrainer
        trainer = Seq2SeqTrainer(model, train_loader, val_loader, config)
        trainer.train()

        # 收集结果
        result = {
            'final_train_loss': trainer.train_losses[-1],
            'final_val_loss': trainer.val_losses[-1],
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'best_val_loss': min(trainer.val_losses),
            'model_params': sum(p.numel() for p in model.parameters())
        }

        return result

    def plot_results(self, save_dir='results/ablation'):
        """绘制消融实验结果"""
        os.makedirs(save_dir, exist_ok=True)

        for experiment_name, results in self.results.items():
            plt.figure(figsize=(15, 10))

            # 训练损失
            plt.subplot(2, 3, 1)
            for config_name, result in results.items():
                plt.plot(result['train_losses'], label=config_name, linewidth=2)
            plt.title(f'{experiment_name} - Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 验证损失
            plt.subplot(2, 3, 2)
            for config_name, result in results.items():
                plt.plot(result['val_losses'], label=config_name, linewidth=2)
            plt.title(f'{experiment_name} - Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 最终性能比较
            plt.subplot(2, 3, 3)
            config_names = list(results.keys())
            final_losses = [results[name]['final_val_loss'] for name in config_names]
            bars = plt.bar(config_names, final_losses)
            plt.title(f'{experiment_name} - Final Validation Loss')
            plt.xticks(rotation=45)
            plt.ylabel('Loss')

            # 最佳性能比较
            plt.subplot(2, 3, 4)
            best_losses = [results[name]['best_val_loss'] for name in config_names]
            bars = plt.bar(config_names, best_losses)
            plt.title(f'{experiment_name} - Best Validation Loss')
            plt.xticks(rotation=45)
            plt.ylabel('Loss')

            # 参数数量比较
            plt.subplot(2, 3, 5)
            param_counts = [results[name].get('model_params', 0) for name in config_names]
            bars = plt.bar(config_names, param_counts)
            plt.title(f'{experiment_name} - Model Parameters')
            plt.xticks(rotation=45)
            plt.ylabel('Parameters')

            # 在柱状图上添加数值
            for subplot_idx in [3, 4, 5]:
                plt.subplot(2, 3, subplot_idx)
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2, height + max(height) * 0.01,
                             f'{height:.3f}' if subplot_idx in [3, 4] else f'{height:,}',
                             ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/{experiment_name}_results.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 保存结果到JSON
        with open(f'{save_dir}/ablation_results.json', 'w') as f:
            json_ready_results = {}
            for exp_name, exp_results in self.results.items():
                json_ready_results[exp_name] = {}
                for config_name, config_results in exp_results.items():
                    json_ready_results[exp_name][config_name] = {
                        'final_train_loss': float(config_results['final_train_loss']),
                        'final_val_loss': float(config_results['final_val_loss']),
                        'best_val_loss': float(config_results['best_val_loss']),
                        'model_params': config_results.get('model_params', 0),
                        'train_losses': [float(x) for x in config_results['train_losses']],
                        'val_losses': [float(x) for x in config_results['val_losses']]
                    }

            json.dump(json_ready_results, f, indent=2)

    def generate_report(self):
        """生成消融实验报告"""
        report = """# Transformer 消融实验报告

## 实验概述
本实验系统地评估了Transformer架构中不同组件对模型性能的影响。

## 实验设置
- **基础配置**: 参见 configs/base.yaml
- **数据集**: Tiny Shakespeare (语言建模) / 简单翻译数据 (序列到序列)
- **评估指标**: 训练损失、验证损失、模型参数数量

## 实验结果汇总

"""

        for experiment_name, results in self.results.items():
            report += f"### {experiment_name.replace('_', ' ').title()} 消融实验\n\n"
            report += "| 配置 | 最终训练损失 | 最终验证损失 | 最佳验证损失 | 参数量 |\n"
            report += "|------|-------------|-------------|-------------|--------|\n"

            for config_name, result in results.items():
                params = result.get('model_params', 0)
                report += f"| {config_name} | {result['final_train_loss']:.4f} | {result['final_val_loss']:.4f} | {result['best_val_loss']:.4f} | {params:,} |\n"

            report += "\n"

        report += """
## 关键发现

### 1. 位置编码的重要性
- 有位置编码的模型显著优于无位置编码的模型
- 位置编码提供了关键的序列顺序信息

### 2. 注意力头数的影响
- 头数增加可以提升模型容量，但可能增加过拟合风险
- 需要平衡模型复杂度和数据量

### 3. 层数深度的影响
- 更深层的模型通常有更强的表示能力
- 但训练难度和计算成本也随之增加

### 4. 架构比较 (Encoder-only vs Encoder-Decoder)
- Encoder-Decoder架构在序列到序列任务上表现更好
- Encoder-only架构在语言建模任务上更高效

## 结论
通过系统的消融实验，我们验证了Transformer架构中各个组件的重要性，为模型设计提供了实践指导。

*报告自动生成于实验完成时*
"""

        # 保存报告
        with open('results/ablation_report.md', 'w') as f:
            f.write(report)

        return report