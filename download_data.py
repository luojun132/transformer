import requests
import os
import torch
from torch.utils.data import Dataset, DataLoader


def download_tiny_shakespeare():
    """手动下载Tiny Shakespeare数据集"""
    print("正在下载Tiny Shakespeare数据集...")

    # 直接从原始仓库下载
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # 检查请求是否成功

        # 创建data目录
        os.makedirs('data', exist_ok=True)

        # 保存数据
        file_path = 'data/tiny_shakespeare.txt'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"数据集已下载到: {file_path}")
        print(f"文件大小: {len(response.text)} 字符")

        return file_path

    except Exception as e:
        print(f"下载失败: {e}")
        return None


def create_simple_tokenizer(texts, vocab_size=5000):
    """创建简单的tokenizer"""
    # 统计词频
    word_freq = {}
    for text in texts:
        words = text.split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

    # 选择最常见的词
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab_words = [word for word, freq in sorted_words[:vocab_size - 4]]  # 保留4个特殊token

    # 创建词汇表
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3
    }

    # 添加普通词汇
    for i, word in enumerate(vocab_words):
        vocab[word] = i + 4

    return vocab


class SimpleTextDataset(Dataset):
    def __init__(self, texts, vocab, max_length):
        self.texts = texts
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        words = text.split()[:self.max_length]

        # 转换为ID
        input_ids = [self.vocab.get(word, self.vocab["[UNK]"]) for word in words]

        # 填充
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.vocab["[PAD]"]] * padding_length

        return torch.tensor(input_ids)


def load_manual_data(batch_size=32, max_length=128):
    """手动加载数据"""
    file_path = 'data/tiny_shakespeare.txt'

    # 如果文件不存在，先下载
    if not os.path.exists(file_path):
        print("数据集文件不存在，开始下载...")
        file_path = download_tiny_shakespeare()
        if file_path is None:
            print("下载失败，使用内置示例数据")
            return create_dummy_data(batch_size, max_length)

    # 读取数据
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 分割成行并清理空行
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    print(f"总共读取 {len(lines)} 行数据")

    # 分割训练集和验证集 (90%训练, 10%验证)
    split_idx = int(0.9 * len(lines))
    train_texts = lines[:split_idx]
    val_texts = lines[split_idx:]

    print(f"训练集: {len(train_texts)} 样本")
    print(f"验证集: {len(val_texts)} 样本")

    # 创建tokenizer
    all_texts = train_texts + val_texts
    vocab = create_simple_tokenizer(all_texts)

    # 创建数据集
    train_dataset = SimpleTextDataset(train_texts, vocab, max_length)
    val_dataset = SimpleTextDataset(val_texts, vocab, max_length)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建简单的tokenizer包装器以保持接口一致
    class SimpleTokenizer:
        def __init__(self, vocab):
            self.vocab = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}

        def get_vocab_size(self):
            return len(self.vocab)

        def token_to_id(self, token):
            return self.vocab.get(token, self.vocab["[UNK]"])

        def encode(self, text):
            words = text.split()
            ids = [self.vocab.get(word, self.vocab["[UNK]"]) for word in words]
            return type('Encoding', (), {'ids': ids})()

        def decode(self, ids):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            tokens = [self.id_to_token.get(i, "[UNK]") for i in ids if i != self.vocab["[PAD]"]]
            return " ".join(tokens)

    tokenizer = SimpleTokenizer(vocab)

    print(f"词汇表大小: {tokenizer.get_vocab_size()}")

    return train_loader, val_loader, tokenizer


def create_dummy_data(batch_size=32, max_length=128):
    """创建虚拟数据作为备选"""
    print("使用虚拟数据进行测试...")

    dummy_texts = [
                      "To be or not to be that is the question",
                      "All the world a stage and all the men and women merely players",
                      "What's in a name that which we call a rose by any other name would smell as sweet",
                      "Some are born great some achieve greatness and some have greatness thrust upon them",
                      "The course of true love never did run smooth",
                      "If music be the food of love play on",
                      "Now is the winter of our discontent made glorious summer by this sun of York",
                      "We are such stuff as dreams are made on and our little life is rounded with a sleep",
                      "The lady doth protest too much methinks",
                      "Brevity is the soul of wit",
                      "Parting is such sweet sorrow",
                      "How sharper than a serpent's tooth it is to have a thankless child",
                      "There are more things in heaven and earth Horatio than are dreamt of in your philosophy",
                      "This above all to thine own self be true",
                      "The fault dear Brutus is not in our stars but in ourselves",
                      "Cowards die many times before their deaths the valiant never taste of death but once",
                      "Men at some time are masters of their fates",
                      "Love looks not with the eyes but with the mind",
                      "The better part of valour is discretion",
                      "All that glitters is not gold"
                  ] * 10  # 重复以增加数据量

    split_idx = int(0.8 * len(dummy_texts))
    train_texts = dummy_texts[:split_idx]
    val_texts = dummy_texts[split_idx:]

    vocab = create_simple_tokenizer(dummy_texts, vocab_size=1000)

    train_dataset = SimpleTextDataset(train_texts, vocab, max_length)
    val_dataset = SimpleTextDataset(val_texts, vocab, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class SimpleTokenizer:
        def __init__(self, vocab):
            self.vocab = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}

        def get_vocab_size(self):
            return len(self.vocab)

        def encode(self, text):
            words = text.split()
            ids = [self.vocab.get(word, self.vocab["[UNK]"]) for word in words]
            return type('Encoding', (), {'ids': ids})()

        def decode(self, ids):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            tokens = [self.id_to_token.get(i, "[UNK]") for i in ids if i != self.vocab["[PAD]"]]
            return " ".join(tokens)

    tokenizer = SimpleTokenizer(vocab)

    print(f"虚拟数据 - 训练集: {len(train_dataset)} 样本")
    print(f"虚拟数据 - 词汇表大小: {tokenizer.get_vocab_size()}")

    return train_loader, val_loader, tokenizer


def create_simple_tokenizer(texts, vocab_size=5000):
    """创建简单的tokenizer - 从之前的代码中提取"""
    # 统计词频
    word_freq = {}
    for text in texts:
        words = text.split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

    # 选择最常见的词
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab_words = [word for word, freq in sorted_words[:vocab_size - 4]]  # 保留4个特殊token

    # 创建词汇表
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3
    }

    # 添加普通词汇
    for i, word in enumerate(vocab_words):
        vocab[word] = i + 4

    return vocab


class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}

    def get_vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        words = text.split()
        ids = [self.vocab.get(word, self.vocab["[UNK]"]) for word in words]
        return type('Encoding', (), {'ids': ids})()

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        tokens = [self.id_to_token.get(i, "[UNK]") for i in ids if i != self.vocab["[PAD]"]]
        return " ".join(tokens)

if __name__ == "__main__":
    # 测试下载和加载
    train_loader, val_loader, tokenizer = load_manual_data()

    # 测试一个样本
    for batch in train_loader:
        print("Batch shape:", batch.shape)
        print("Sample text:", tokenizer.decode(batch[0]))
        break