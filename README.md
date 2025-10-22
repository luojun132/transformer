# Transformer from Scratch

A complete implementation of Transformer architecture from scratch with training pipeline.

## Features

- Multi-head self-attention
- Position-wise feed-forward networks
- Residual connections + Layer normalization
- Sinusoidal positional encoding
- Training stability techniques (AdamW, gradient clipping, learning rate scheduling)
- Visualization and model saving

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/transformer-from-scratch.git
cd transformer-from-scratch

# Run training script
chmod +x scripts/run.sh
./scripts/run.sh

# Or run manually
pip install -r requirements.txt
python train.py