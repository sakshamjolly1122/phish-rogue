#!/usr/bin/env bash
set -e

echo "🚀 Starting Stage-A training and inference..."

# Train Stage-A model
echo "📚 Training Stage-A model..."
python3 -m phish_rogue.train_stage_a --config configs/default.yaml

# Run Stage-A inference on validation set
echo "🔍 Running Stage-A inference on validation set..."
python3 -m phish_rogue.infer_stage_a --config configs/default.yaml --split val

# Route uncertain samples
echo "🎯 Routing uncertain samples..."
python3 -m phish_rogue.route_uncertainty --config configs/default.yaml \
  --input outputs/stage_a/preds_val.csv \
  --out outputs/stage_a/val_routed.csv

echo "✅ Stage-A pipeline completed!"
echo "📁 Check outputs/stage_a/ for results"
