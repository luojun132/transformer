import torch
import yaml
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import TransformerLM
from download_data import load_manual_data


def quick_model_test():
    """快速测试模型是否能正常运行"""
    print("=== 快速模型测试 ===")

    try:
        # 加载小批量数据
        train_loader, val_loader, tokenizer = load_manual_data(
            batch_size=8,  # 小batch
            max_length=64  # 短序列
        )

        # 测试模型初始化
        model = TransformerLM(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=128,
            num_heads=4,
            d_ff=256,  # 较小的FFN
            num_layers=2,
            max_seq_len=64,
            dropout=0.1,
            use_positional_encoding=True
        )

        print(f"✅ 模型初始化成功!")
        print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

        # 测试前向传播
        for batch in train_loader:
            print(f"输入形状: {batch.shape}")

            # 准备输入和目标
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            # 前向传播
            with torch.no_grad():
                logits = model(input_ids)
                print(f"输出logits形状: {logits.shape}")

                # 计算损失
                criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1)
                )
                print(f"初始损失: {loss.item():.4f}")

            break  # 只测试一个batch

        print("✅ 模型前向传播测试通过!")
        return True

    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False


def test_positional_encoding():
    """测试位置编码功能"""
    print("\n=== 位置编码测试 ===")

    try:
        # 测试有位置编码
        model_with_pe = TransformerLM(
            vocab_size=1000,
            d_model=64,
            num_heads=2,
            d_ff=128,
            num_layers=1,
            max_seq_len=32,
            dropout=0.1,
            use_positional_encoding=True
        )
        print("✅ 有位置编码模型创建成功")

        # 测试无位置编码
        model_without_pe = TransformerLM(
            vocab_size=1000,
            d_model=64,
            num_heads=2,
            d_ff=128,
            num_layers=1,
            max_seq_len=32,
            dropout=0.1,
            use_positional_encoding=False
        )
        print("✅ 无位置编码模型创建成功")

        # 创建测试输入
        test_input = torch.randint(0, 1000, (2, 16))  # batch_size=2, seq_len=16

        # 前向传播
        with torch.no_grad():
            output_with_pe = model_with_pe(test_input)
            output_without_pe = model_without_pe(test_input)

            print(f"有位置编码输出形状: {output_with_pe.shape}")
            print(f"无位置编码输出形状: {output_without_pe.shape}")

            # 检查输出是否不同
            diff = torch.abs(output_with_pe - output_without_pe).mean()
            print(f"输出差异: {diff.item():.6f}")

            if diff > 1e-6:
                print("✅ 位置编码正常工作!")
            else:
                print("❌ 位置编码可能没有生效")

        return True

    except Exception as e:
        print(f"❌ 位置编码测试失败: {e}")
        return False


def test_all_components():
    """测试所有组件"""
    print("=== 开始全面测试 ===")

    tests_passed = 0
    total_tests = 2

    # 测试1: 基础模型
    if quick_model_test():
        tests_passed += 1

    # 测试2: 位置编码
    if test_positional_encoding():
        tests_passed += 1

    print(f"\n=== 测试结果: {tests_passed}/{total_tests} 通过 ===")

    if tests_passed == total_tests:
        print("🎉 所有测试通过! 可以开始正式训练。")
        return True
    else:
        print("❌ 部分测试失败，请检查代码。")
        return False


if __name__ == "__main__":
    # 直接运行测试函数，不使用pytest
    test_all_components()