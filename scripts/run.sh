#!/bin/bash

# Set environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create required directories
mkdir -p checkpoints results

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

echo "Training completed!"