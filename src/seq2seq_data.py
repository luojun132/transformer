import torch
from torch.utils.data import Dataset, DataLoader
import os
import requests


class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_length):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # 编码源文本和目标文本
        src_encoding = self.src_tokenizer.encode(src_text)
        tgt_encoding = self.tgt_tokenizer.encode(tgt_text)

        # 截断和填充
        src_ids = src_encoding.ids[:self.max_length]
        tgt_ids = tgt_encoding.ids[:self.max_length - 1]  # 留一个位置给起始符

        src_padding = self.max_length - len(src_ids)
        tgt_padding = self.max_length - len(tgt_ids) - 1  # -1 因为decoder输入要移位

        if src_padding > 0:
            src_ids = src_ids + [self.src_tokenizer.token_to_id("[PAD]")] * src_padding

        # decoder输入 (shifted right)
        decoder_input_ids = [self.tgt_tokenizer.token_to_id("[BOS]")] + tgt_ids
        if tgt_padding > 0:
            decoder_input_ids = decoder_input_ids + [self.tgt_tokenizer.token_to_id("[PAD]")] * tgt_padding

        # labels (shifted left)
        label_ids = tgt_ids + [self.tgt_tokenizer.token_to_id("[EOS]")]
        if tgt_padding > 0:
            label_ids = label_ids + [self.tgt_tokenizer.token_to_id("[PAD]")] * (tgt_padding - 1)

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(decoder_input_ids, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long)
        )


def create_simple_translation_data():
    """创建简单的英法翻译数据用于演示"""
    english_texts = [
        "Hello world",
        "How are you",
        "I am fine",
        "What is your name",
        "My name is John",
        "Where are you from",
        "I am from China",
        "How old are you",
        "I am twenty years old",
        "What time is it",
        "It is nine o clock",
        "Thank you very much",
        "You are welcome",
        "I love programming",
        "This is a test",
        "The weather is nice",
        "I like to learn",
        "Machine learning is interesting",
        "Deep learning models",
        "Transformers are powerful"
    ]

    french_texts = [
        "Bonjour le monde",
        "Comment allez vous",
        "Je vais bien",
        "Comment vous appelez vous",
        "Je m appelle Jean",
        "D où venez vous",
        "Je viens de Chine",
        "Quel âge avez vous",
        "J ai vingt ans",
        "Quelle heure est il",
        "Il est neuf heures",
        "Merci beaucoup",
        "De rien",
        "J aime la programmation",
        "Ceci est un test",
        "Le temps est beau",
        "J aime apprendre",
        "L apprentissage automatique est intéressant",
        "Modèles d apprentissage profond",
        "Les transformateurs sont puissants"
    ]

    return english_texts, french_texts


def load_translation_data(batch_size=32, max_length=128):
    """加载翻译数据"""
    print("Loading translation data...")

    # 创建简单的翻译数据
    src_texts, tgt_texts = create_simple_translation_data()

    # 重复数据以增加数量
    src_texts = src_texts * 10
    tgt_texts = tgt_texts * 10

    # 创建tokenizer - 确保使用SimpleTokenizer类
    from download_data import create_simple_tokenizer, SimpleTokenizer

    src_vocab = create_simple_tokenizer(src_texts, vocab_size=2000)
    tgt_vocab = create_simple_tokenizer(tgt_texts, vocab_size=2000)

    # 创建SimpleTokenizer对象
    src_tokenizer = SimpleTokenizer(src_vocab)
    tgt_tokenizer = SimpleTokenizer(tgt_vocab)

    # 分割训练集和验证集
    split_idx = int(0.8 * len(src_texts))

    train_dataset = TranslationDataset(
        src_texts[:split_idx], tgt_texts[:split_idx],
        src_tokenizer, tgt_tokenizer, max_length
    )

    val_dataset = TranslationDataset(
        src_texts[split_idx:], tgt_texts[split_idx:],
        src_tokenizer, tgt_tokenizer, max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Source vocabulary: {src_tokenizer.get_vocab_size()}")
    print(f"Target vocabulary: {tgt_tokenizer.get_vocab_size()}")

    return train_loader, val_loader, src_tokenizer, tgt_tokenizer