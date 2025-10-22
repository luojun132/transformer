import os
import glob
import torch
from datetime import datetime


def check_saved_checkpoints():
    """检查已保存的模型检查点"""
    print("=== 检查模型保存情况 ===")

    checkpoint_files = glob.glob('checkpoints/*.pt')

    if not checkpoint_files:
        print("❌ 没有找到任何检查点文件")
        return False

    print(f"✅ 找到 {len(checkpoint_files)} 个检查点文件:")

    for checkpoint_file in checkpoint_files:
        file_size = os.path.getsize(checkpoint_file) / 1024 / 1024  # MB
        file_time = datetime.fromtimestamp(os.path.getctime(checkpoint_file))

        try:
            # 尝试读取检查点基本信息
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            epoch = checkpoint.get('epoch', '未知')
            exp_name = checkpoint.get('experiment_name', '未知实验')

            if 'train_losses' in checkpoint:
                train_loss = checkpoint['train_losses'][-1] if checkpoint['train_losses'] else 'N/A'
                val_loss = checkpoint['val_losses'][-1] if checkpoint['val_losses'] else 'N/A'
                print(f"   📁 {os.path.basename(checkpoint_file)}")
                print(f"     实验: {exp_name}, Epoch: {epoch}")
                print(f"     训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
                print(f"     文件大小: {file_size:.1f} MB, 保存时间: {file_time.strftime('%H:%M:%S')}")
            else:
                print(f"   ⚠️ {checkpoint_file} - 文件格式异常")

        except Exception as e:
            print(f"   ❌ {checkpoint_file} - 读取失败: {e}")

        print()  # 空行分隔

    return True


def check_directory_structure():
    """检查目录结构"""
    print("\n=== 目录结构检查 ===")

    directories = ['checkpoints', 'results', 'results/ablation', 'data']

    for dir_path in directories:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"✅ {dir_path}: {len(files)} 个文件/目录")
            if files:
                for file in files[:3]:  # 显示前3个
                    print(f"   - {file}")
                if len(files) > 3:
                    print(f"   ... 还有 {len(files) - 3} 个文件")
        else:
            print(f"❌ {dir_path}: 目录不存在")


if __name__ == "__main__":
    has_checkpoints = check_saved_checkpoints()
    check_directory_structure()

    if has_checkpoints:
        print("\n🎉 模型正在正常保存！训练可以继续。")
    else:
        print("\n⚠️ 未找到检查点，可能需要检查保存逻辑。")