import yaml
import torch
from src.experiment import AblationExperiment

def main():
    # 加载基础配置 - 指定UTF-8编码
    with open('configs/base.yaml', 'r', encoding='utf-8') as f:  # 添加encoding='utf-8'
        base_config = yaml.safe_load(f)

    # 大幅减少epochs用于快速实验
    base_config['num_epochs'] = 8  # 从15减少到8
    base_config['batch_size'] = 16
    base_config['learning_rate'] = 0.002  # 稍微提高学习率加速收敛

    # 设置随机种子
    torch.manual_seed(base_config['random_seed'])

    # 创建实验管理器
    experiment = AblationExperiment(base_config)

    print("Starting fast ablation studies...")
    print(f"配置: {base_config['num_epochs']} epochs, batch_size={base_config['batch_size']}")

    # 运行所有消融实验
    print("1. Running positional encoding ablation...")
    experiment.run_positional_encoding_ablation()

    print("2. Running heads ablation...")
    experiment.run_heads_ablation()

    print("3. Running layers ablation...")
    experiment.run_layers_ablation()

    # 生成结果和报告
    print("Generating results and reports...")
    experiment.plot_results()
    report = experiment.generate_report()

    print("Ablation studies completed!")
    print("Results saved to results/ablation/")
    print("\nReport Summary:")
    print(report)

if __name__ == "__main__":
    main()