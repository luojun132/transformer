import torch
import yaml
from src.model import TransformerSeq2Seq
from src.seq2seq_data import load_translation_data
from src.seq2seq_trainer import Seq2SeqTrainer


def main():
    # 加载配置
    with open('configs/seq2seq.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 设置随机种子
    torch.manual_seed(config['random_seed'])

    print("Loading translation data...")
    train_loader, val_loader, src_tokenizer, tgt_tokenizer = load_translation_data(
        batch_size=config['batch_size'],
        max_length=config['max_seq_len']
    )

    print("Initializing Seq2Seq Transformer...")
    model = TransformerSeq2Seq(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        use_positional_encoding=config['use_positional_encoding']
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 训练
    trainer = Seq2SeqTrainer(model, train_loader, val_loader, config)
    trainer.train()


if __name__ == "__main__":
    main()