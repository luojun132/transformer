import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import os
import time


class TransformerTrainer:
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

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)

            # Prepare input and target
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(input_ids)

            # Calculate loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                input_ids = batch[:, :-1]
                target_ids = batch[:, 1:]

                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1)
                )
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, experiment_name=""):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'experiment_name': experiment_name
        }

        os.makedirs('checkpoints', exist_ok=True)

        # 添加实验名称到文件名，避免覆盖
        if experiment_name:
            filename = f'checkpoints/checkpoint_{experiment_name}_epoch_{epoch}.pt'
        else:
            filename = f'checkpoints/checkpoint_epoch_{epoch}.pt'

        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    def train(self):
        print("Starting training...")

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

            # 改进的保存逻辑：每个epoch都保存，但使用不同的策略
            exp_name = self.config.get('experiment_name', 'unknown')

            # 策略1：每个epoch都保存轻量级检查点（只保存最新）
            if epoch == self.config['num_epochs'] - 1:  # 最后一个epoch
                self.save_checkpoint(epoch, f"{exp_name}_final")

            # 策略2：每2个epoch保存完整检查点
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint(epoch, f"{exp_name}_epoch_{epoch + 1}")

            # 策略3：如果性能提升明显，也保存
            if len(self.val_losses) > 1 and val_loss < min(self.val_losses[:-1]):
                self.save_checkpoint(epoch, f"{exp_name}_best")

        self.plot_training_curve()

    def plot_training_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        os.makedirs('results', exist_ok=True)
        plt.savefig('results/training_curve.png')
        plt.close()