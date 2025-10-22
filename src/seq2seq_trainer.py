import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import os
import time


class Seq2SeqTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

        # Training history
        self.train_losses = []
        self.val_losses = []

    def create_masks(self, src, tgt):
        """创建源掩码和目标掩码"""
        # 源掩码 (padding mask)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # 目标掩码 (padding mask + look-ahead mask)
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        tgt_look_ahead_mask = tgt_look_ahead_mask.to(self.device)
        tgt_mask = tgt_padding_mask & ~tgt_look_ahead_mask.unsqueeze(0).unsqueeze(0)

        return src_mask, tgt_mask

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch_idx, (src, tgt_input, tgt_output) in enumerate(self.train_loader):
            src = src.to(self.device)
            tgt_input = tgt_input.to(self.device)
            tgt_output = tgt_output.to(self.device)

            # 创建掩码
            src_mask, tgt_mask = self.create_masks(src, tgt_input)

            self.optimizer.zero_grad()

            # 前向传播
            logits = self.model(src, tgt_input, src_mask, tgt_mask)

            # 计算损失
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for src, tgt_input, tgt_output in self.val_loader:
                src = src.to(self.device)
                tgt_input = tgt_input.to(self.device)
                tgt_output = tgt_output.to(self.device)

                src_mask, tgt_mask = self.create_masks(src, tgt_input)

                logits = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1)
                )
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        print("Starting sequence-to-sequence training...")

        for epoch in range(self.config['num_epochs']):
            start_time = time.time()

            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            epoch_time = time.time() - start_time

            print(f'Epoch {epoch + 1}/{self.config["num_epochs"]}, '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Time: {epoch_time:.2f}s')

            # 保存模型检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

        self.plot_training_curve()

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/seq2seq_checkpoint_epoch_{epoch}.pt')

    def plot_training_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Sequence-to-Sequence Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        os.makedirs('results', exist_ok=True)
        plt.savefig('results/seq2seq_training_curve.png')
        plt.close()