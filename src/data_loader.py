import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

# 设置Hugging Face国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode(text)

        # Truncate or pad to max_length
        input_ids = encoding.ids[:self.max_length]
        padding_length = self.max_length - len(input_ids)

        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.token_to_id("[PAD]")] * padding_length

        return torch.tensor(input_ids)


def create_tokenizer(texts, vocab_size=10000):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    )

    # Train tokenizer
    def get_training_corpus():
        for i in range(0, len(texts), 1000):
            yield texts[i:i + 1000]

    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    return tokenizer


def load_tiny_shakespeare(batch_size=32, max_length=128):
    """Load Tiny Shakespeare dataset"""
    dataset = load_dataset("tiny_shakespeare")

    # Combine all text
    all_text = dataset['train']['text']

    # Create tokenizer
    tokenizer = create_tokenizer(all_text)

    # Create datasets
    train_dataset = TextDataset(dataset['train']['text'], tokenizer, max_length)
    val_dataset = TextDataset(dataset['validation']['text'], tokenizer, max_length)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer