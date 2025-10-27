#!/usr/bin/env bash
set -e

echo "🚀 Starting Stage-B training and end-to-end inference..."

# Train Stage-B model
echo "📚 Training Stage-B fusion model..."
python3 -m phish_rogue.train_stage_b --config configs/default.yaml \
  --routed_csv outputs/stage_a/val_routed.csv

# Run end-to-end inference on test set
echo "🔍 Running end-to-end inference on test set..."
python3 -m phish_rogue.infer_pipeline --config configs/default.yaml --split test

echo "✅ Stage-B pipeline completed!"
echo "📁 Check outputs/ for final results"
echo "📊 Check outputs/plots/ for visualizations"
