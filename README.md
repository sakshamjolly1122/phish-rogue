# PHISH-ROGUE

**Two-stage, uncertainty-aware phishing and malicious URL detector combining fast char-level deep learning with adaptive HTML feature analysis.**

## 🎯 Overview

PHISH-ROGUE is an innovative two-stage phishing URL detection system that combines:

- **Stage-A**: Fast char-level CNN + Transformer URL classifier for initial screening
- **Stage-B**: Lightweight HTML content feature analysis for uncertain samples
- **Uncertainty-aware routing**: Only escalates samples when Stage-A is uncertain
- **Adversarial robustness**: URL augmentation techniques for better generalization
- **Multi-class support**: Handles benign, phishing, malware, spam, and defacement URLs

## 📁 Project Structure

```
340WW/
├── raw/                          # Raw CSV data files
│   ├── train (1).csv
│   ├── val_unseen (1).csv
│   └── test (2).csv
├── data/processed/               # Processed data files
├── configs/                      # Configuration files
│   └── default.yaml
├── phish_rogue/                  # Core Python package
│   ├── __init__.py
│   ├── utils.py                  # General utilities
│   ├── tokenizer.py              # Char-level tokenization
│   ├── augment.py                # URL augmentation
│   ├── dataio.py                 # Data loading
│   ├── metrics.py                # Evaluation metrics
│   ├── model_stage_a.py          # Stage-A CNN+Transformer
│   ├── train_stage_a.py          # Stage-A training
│   ├── infer_stage_a.py          # Stage-A inference
│   ├── route_uncertainty.py      # Uncertainty routing
│   ├── features_content.py       # HTML feature extraction
│   ├── model_stage_b.py          # Stage-B fusion model
│   ├── train_stage_b.py          # Stage-B training
│   └── infer_pipeline.py         # End-to-end inference
├── scripts/                      # Utility scripts
│   ├── prepare_data.py
│   ├── run_stage_a.sh
│   └── run_stage_b.sh
├── outputs/                      # Model outputs and results
│   ├── stage_a/                  # Stage-A checkpoints and results
│   ├── stage_b/                  # Stage-B checkpoints and results
│   └── plots/                    # Visualization plots
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd ~/Desktop/340WW
python -m pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python scripts/prepare_data.py
```

This copies and renames your raw CSV files to the processed directory.

### 3. Run Complete Pipeline

```bash
# Stage-A: Train model and run inference
bash scripts/run_stage_a.sh

# Stage-B: Train fusion model and run end-to-end inference
bash scripts/run_stage_b.sh
```

## 📊 Expected Outputs

After running the complete pipeline, you'll find:

### Stage-A Outputs (`outputs/stage_a/`)
- `best.pt` - Best Stage-A model checkpoint
- `tokenizer.json` - Character tokenizer
- `label_mappings.json` - Label to ID mappings
- `metrics.json` - Training metrics
- `preds_val.csv` - Validation predictions
- `embeddings_val.npy` - URL embeddings
- `val_routed.csv` - Routed samples with uncertainty flags

### Stage-B Outputs (`outputs/stage_b/`)
- `best.pt` - Best Stage-B fusion model checkpoint
- `content_feat_names.json` - HTML feature names
- `metrics_val.json` - Stage-B training metrics

### Final Results (`outputs/`)
- `final_preds_test.csv` - Final predictions with Stage-A/Stage-B breakdown
- `reports_test.json` - Comprehensive metrics report
- `plots/` - Confusion matrix and reliability diagrams

## ⚙️ Configuration

Edit `configs/default.yaml` to customize:

- **Task type**: Switch between `multiclass` and `binary`
- **Model architecture**: Adjust Stage-A CNN/Transformer parameters
- **Training**: Learning rates, batch sizes, epochs
- **Routing thresholds**: Uncertainty and confidence thresholds
- **HTML features**: Timeout, feature dimensions

### Example: Binary Classification

```yaml
task_type: "binary"
class_names: ["benign", "malicious"]
```

## 🔬 Key Features

### Stage-A: Fast URL Classification
- **Char-level CNN**: Captures local URL patterns
- **Transformer**: Models long-range dependencies
- **Lightweight**: ~150 vocab size, fast inference
- **Robust**: Adversarial URL augmentation

### Stage-B: HTML Content Analysis
- **Adaptive**: Only processes uncertain samples
- **Lightweight**: 20-dimensional HTML features
- **Fast**: 150ms timeout per URL
- **Comprehensive**: Title, forms, scripts, links, etc.

### Uncertainty-Aware Routing
- **Entropy thresholding**: High uncertainty samples
- **Class-specific confidence**: Different thresholds per class
- **Cost-aware**: Balances accuracy vs. computational cost

## 📈 Performance Metrics

The system provides comprehensive evaluation:

- **Accuracy**: Overall classification accuracy
- **F1 Scores**: Per-class and macro F1
- **Confusion Matrix**: Detailed classification breakdown
- **Expected Calibration Error (ECE)**: Model confidence calibration
- **Escalation Rate**: Percentage of samples sent to Stage-B

## 🛠️ Advanced Usage

### Custom Training

```bash
# Train Stage-A only
python -m phish_rogue.train_stage_a --config configs/default.yaml

# Train Stage-B with custom routed data
python -m phish_rogue.train_stage_b --config configs/default.yaml \
  --routed_csv custom_routed.csv
```

### Custom Inference

```bash
# Run Stage-A inference only
python -m phish_rogue.infer_stage_a --config configs/default.yaml --split test

# Run end-to-end pipeline
python -m phish_rogue.infer_pipeline --config configs/default.yaml --split test
```

### Uncertainty Routing

```bash
# Route samples with custom thresholds
python -m phish_rogue.route_uncertainty --config configs/default.yaml \
  --input predictions.csv --out routed.csv
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in config
2. **HTML fetch timeouts**: Increase `html_timeout_ms` in config
3. **Missing dependencies**: Run `pip install -r requirements.txt`
4. **Data format errors**: Ensure CSV has `url` and `label` columns

### Performance Tips

- Use GPU for faster training (`torch.cuda.is_available()`)
- Adjust `augment_prob` for more/fewer augmented samples
- Tune routing thresholds based on your accuracy/speed requirements

## 🔮 Future Work

- **Graph features**: Add URL graph-based features
- **Temperature scaling**: Improve calibration
- **Ensemble methods**: Combine multiple models
- **Real-time deployment**: Optimize for production use

## 📝 Citation

If you use PHISH-ROGUE in your research, please cite:

```bibtex
@article{phish_rogue_2024,
  title={PHISH-ROGUE: Two-Stage Uncertainty-Aware Phishing URL Detection},
  author={PHISH-ROGUE Team},
  journal={arXiv preprint},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for cybersecurity research and education.**
