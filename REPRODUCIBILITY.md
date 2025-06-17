# TW-MRT Reproducibility Guide

This document provides comprehensive instructions for reproducing the results from our paper:

> **Trend-Weighted Multi-Resolution Transformer for Multi-Parametric Glucose Prediction**

## 📋 Table of Contents

1. [Quick Start with Docker (Recommended)](#quick-start-with-docker)
2. [Manual Setup](#manual-setup)
3. [Configuration Management](#configuration-management)
4. [Training and Logging](#training-and-logging)
5. [Results Analysis](#results-analysis)
6. [Troubleshooting](#troubleshooting)

## 🐳 Quick Start with Docker (Recommended)

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)

### Step 1: Clone and Setup
```bash
git clone https://github.com/your-username/glucose-prediction-TW-MRT.git
cd glucose-prediction-TW-MRT
```

### Step 2: Build and Run
```bash
# Build the Docker image
docker-compose build

# Run the training (CPU)
docker-compose run tw-mrt python main_reproducible.py

# Run the training (GPU - requires nvidia-docker)
docker-compose run tw-mrt python main_reproducible.py --gpu 0

# Run TensorBoard for monitoring
docker-compose up tensorboard
```

### Step 3: Access TensorBoard
Visit http://localhost:6006 to view training progress and logs.

## 🔧 Manual Setup

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU support)
- Git

### Step 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/glucose-prediction-TW-MRT.git
cd glucose-prediction-TW-MRT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Dataset Preparation
1. Download the OhioT1DM dataset from [official source]
2. Place the data files in the `data/` directory
3. Ensure the following structure:
```
data/
├── patient552/
│   ├── glucose_data.csv
│   └── ...
└── patient575/
    ├── glucose_data.csv
    └── ...
```

### Step 3: Run Training
```bash
# Basic training
python main_reproducible.py

# Custom configuration
python main_reproducible.py --config custom_config.yaml

# GPU training
python main_reproducible.py --gpu 0

# Dry run (testing)
python main_reproducible.py --dry-run
```

## ⚙️ Configuration Management

All hyperparameters and experiment settings are managed through the `config.yaml` file:

### Key Configuration Sections

#### Model Architecture
```yaml
model:
  d_model: 64          # Transformer dimension
  nhead: 4             # Number of attention heads
  num_layers: 3        # Number of transformer layers
  dropout: 0.1         # Dropout rate
```

#### Training Settings
```yaml
training:
  num_epochs: 100      # Maximum epochs
  learning_rate: 0.001 # Initial learning rate
  batch_size: 32       # Batch size
  optimizer: "AdamW"   # Optimizer type
```

#### Experiment Parameters
```yaml
experiment_params:
  input_windows: [3]         # Input window sizes (×5 min)
  prediction_horizons: [12]  # Prediction horizons (×5 min)
  feature_numbers: [3]       # Number of input features
```

### Creating Custom Configurations

1. Copy `config.yaml` to `my_config.yaml`
2. Modify parameters as needed
3. Run with: `python main_reproducible.py --config my_config.yaml`

## 📊 Training and Logging

### Comprehensive Logging Features

Our logging system captures:
- **System Information**: Hardware, software versions
- **Training Progress**: Loss curves, learning rates, metrics
- **Model Checkpoints**: Best models saved automatically
- **Predictions**: Raw predictions and ground truth values
- **Configuration**: Complete parameter snapshot

### Log Output Structure
```
logs/
├── training_20231215_143022.log     # Detailed text logs
├── tensorboard_20231215_143022/     # TensorBoard logs
│   ├── events.out.tfevents...
│   └── ...
└── ...

results/
├── config_20231215_143022.yaml      # Used configuration
├── results_20231215_143022.json     # Complete results
├── model_20231215_143022.pth        # Model checkpoint
├── patient552_model.pth             # Patient-specific models
├── patient552_actual_values.csv     # Ground truth
├── patient552_predicted_values.csv  # Predictions
└── glucose_prediction_results_*.xlsx # Excel reports
```

### Monitoring Training

#### Real-time Monitoring
```bash
# Start TensorBoard
tensorboard --logdir=logs/

# Or with Docker
docker-compose up tensorboard
```

#### Log Analysis
```bash
# View latest log
tail -f logs/training_*.log

# Search for specific metrics
grep "RMSE" logs/training_*.log
```

## 📈 Results Analysis

### Generated Output Files

1. **Excel Reports**: Detailed metrics per patient and configuration
2. **Model Checkpoints**: Trained models for inference
3. **Prediction Data**: CSV files with predictions vs. actual values
4. **Training Logs**: Complete training history
5. **Configuration Snapshots**: Exact parameters used

### Key Metrics Reported

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination

### Statistical Analysis

The system automatically computes:
- Per-patient metrics
- Cross-patient averages
- Statistical significance tests
- Confidence intervals

## 🔍 Reproducibility Features

### Deterministic Training
- Fixed random seeds across all components
- Deterministic CUDA operations
- Reproducible data loading order

### Version Control
- Pinned dependency versions
- Docker images for environment consistency
- Git commit hashes in logs

### Complete Experiment Tracking
- All hyperparameters logged
- System information captured
- Random states saved
- Complete code snapshots

## 🐛 Troubleshooting

### Common Issues

#### Out of Memory (GPU)
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 16  # Reduce from 32
```

#### CUDA Not Available
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Run on CPU
python main_reproducible.py --gpu -1
```

#### Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

#### Data Loading Issues
```bash
# Check data directory structure
ls -la data/

# Run with verbose logging
python main_reproducible.py --config config.yaml
```

### Performance Optimization

#### Memory Usage
- Monitor GPU memory: `nvidia-smi`
- Reduce batch size if needed
- Use gradient checkpointing for large models

#### Training Speed
- Use multiple GPUs with DataParallel
- Enable mixed precision training
- Optimize data loading with more workers

## 📞 Support

### Getting Help

1. **Check Logs**: Always check the detailed logs first
2. **Issue Tracker**: Report bugs on GitHub
3. **Documentation**: Refer to code comments and docstrings
4. **Contact**: Email the authors for research questions

### Citing This Work

If you use this code, please cite our paper:

```bibtex
@article{your_paper_2024,
  title={Trend-Weighted Multi-Resolution Transformer for Multi-Parametric Glucose Prediction},
  author={Your Name et al.},
  journal={Your Journal},
  year={2024}
}
```

## 📄 License

This code is provided for academic research purposes. Please see LICENSE file for details.

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Maintainer**: [Your Name] <your.email@institution.edu> 