#!/bin/bash

echo "Installing Healthcare Fraud Detection Dependencies..."
echo

echo "Step 1: Upgrading pip..."
python -m pip install --upgrade pip
echo

echo "Step 2: Installing core dependencies..."
pip install -r requirements.txt
echo

echo "Step 3: Installing PyTorch..."
pip install torch torchvision torchaudio
echo

echo "Step 4: Installing PyTorch Geometric and dependencies..."
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
echo

echo "Step 5: Installing optional Neo4j (if needed)..."
# Uncomment the next line if you need Neo4j
# pip install neo4j
echo

echo
echo "Installation complete!"

