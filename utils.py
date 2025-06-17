import os
import json
import yaml
import logging
import datetime
import torch
import numpy as np
from typing import Dict, Any, List
from torch.utils.tensorboard import SummaryWriter

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup comprehensive logging system."""
    log_dir = config['experiment']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for unique log files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('TW-MRT')
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def setup_tensorboard(config: Dict[str, Any]) -> SummaryWriter:
    """Setup TensorBoard logging."""
    if not config['logging']['tensorboard']:
        return None
    
    log_dir = config['experiment']['log_dir']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = os.path.join(log_dir, f"tensorboard_{timestamp}")
    
    writer = SummaryWriter(tb_dir)
    return writer

def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to file for reproducibility."""
    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def set_deterministic_mode(config: Dict[str, Any]):
    """Set deterministic mode for reproducibility."""
    if config['reproducibility']['deterministic']:
        torch.manual_seed(config['experiment']['seed'])
        np.random.seed(config['experiment']['seed'])
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['experiment']['seed'])
            torch.cuda.manual_seed_all(config['experiment']['seed'])
        
        torch.backends.cudnn.deterministic = config['hardware']['cuda_deterministic']
        torch.backends.cudnn.benchmark = config['hardware']['cuda_benchmark']
        
        if config['reproducibility']['use_deterministic_algorithms']:
            torch.use_deterministic_algorithms(True, warn_only=config['reproducibility']['warn_only'])

class TrainingLogger:
    """Comprehensive training logger with metrics tracking."""
    
    def __init__(self, config: Dict[str, Any], writer: SummaryWriter = None):
        self.config = config
        self.writer = writer
        self.logger = logging.getLogger('TW-MRT.Training')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        self.start_time = None
        
    def start_training(self):
        """Mark the start of training."""
        self.start_time = datetime.datetime.now()
        self.logger.info("Training started")
        self.log_system_info()
        self.log_config()
    
    def log_system_info(self):
        """Log system and hardware information."""
        self.logger.info("=== System Information ===")
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            self.logger.info(f"GPU count: {torch.cuda.device_count()}")
            self.logger.info(f"GPU name: {torch.cuda.get_device_name()}")
        self.logger.info("========================")
    
    def log_config(self):
        """Log configuration details."""
        self.logger.info("=== Configuration ===")
        self.logger.info(f"Experiment: {self.config['experiment']['name']}")
        self.logger.info(f"Model: {self.config['model']['name']}")
        self.logger.info(f"Learning rate: {self.config['training']['learning_rate']}")
        self.logger.info(f"Batch size: {self.config['data']['batch_size']}")
        self.logger.info(f"Max epochs: {self.config['training']['num_epochs']}")
        self.logger.info("====================")
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  lr: float, epoch_time: float, metrics: Dict[str, float] = None):
        """Log epoch results."""
        # Store in history
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['learning_rate'].append(lr)
        self.training_history['epoch_time'].append(epoch_time)
        
        # Console logging
        self.logger.info(
            f"Epoch {epoch:3d}/{self.config['training']['num_epochs']:3d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {lr:.2e} | "
            f"Time: {epoch_time:.2f}s"
        )
        
        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', lr, epoch)
            self.writer.add_scalar('Time/Epoch', epoch_time, epoch)
            
            if metrics:
                for name, value in metrics.items():
                    self.writer.add_scalar(f'Metrics/{name}', value, epoch)
    
    def log_final_results(self, test_metrics: Dict[str, float]):
        """Log final test results."""
        total_time = (datetime.datetime.now() - self.start_time).total_seconds()
        
        self.logger.info("=== Training Completed ===")
        self.logger.info(f"Total training time: {total_time/60:.2f} minutes")
        self.logger.info("=== Test Results ===")
        for metric, value in test_metrics.items():
            self.logger.info(f"{metric}: {value:.6f}")
        self.logger.info("====================")
        
        if self.writer:
            for metric, value in test_metrics.items():
                self.writer.add_scalar(f'Test/{metric}', value)
    
    def save_history(self, save_path: str):
        """Save training history to file."""
        with open(save_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def close(self):
        """Close logger resources."""
        if self.writer:
            self.writer.close()

def save_experiment_results(config: Dict[str, Any], results: Dict[str, Any], 
                          model_state: Dict[str, Any] = None):
    """Save complete experiment results."""
    output_dir = config['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save configuration
    config_path = os.path.join(output_dir, f"config_{timestamp}.yaml")
    save_config(config, config_path)
    
    # Save results
    results_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model if provided
    if model_state and config['logging']['save_model']:
        model_path = os.path.join(output_dir, f"model_{timestamp}.pth")
        torch.save(model_state, model_path)
    
    return {
        'config_path': config_path,
        'results_path': results_path,
        'model_path': model_path if model_state else None
    } 