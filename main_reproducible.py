#!/usr/bin/env python3
"""
Reproducible Training Script for TW-MRT Model
Trend-Weighted Multi-Resolution Transformer for Glucose Prediction

This script provides full reproducibility with:
- Configuration-based parameter management
- Comprehensive logging
- Deterministic training
- Experiment tracking
"""

import os
import sys
import argparse
import time
import xlsxwriter
import pandas as pd
import torch
from utils import (
    load_config, setup_logging, setup_tensorboard, 
    set_deterministic_mode, TrainingLogger, save_experiment_results
)
from testModel import test_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TW-MRT Glucose Prediction Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Override experiment name from config')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory from config')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID to use')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without actual training (for testing)')
    return parser.parse_args()

def setup_environment(config, args):
    """Setup the training environment."""
    # Set GPU device
    gpu_id = args.gpu if args.gpu is not None else config['hardware']['cuda_visible_devices']
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Set deterministic mode
    set_deterministic_mode(config)
    
    # Create output directories
    output_dir = args.output_dir or config['experiment']['output_dir']
    log_dir = config['experiment']['log_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    return output_dir, log_dir

def run_single_experiment(config, patient_index, patient, feature_number, 
                         prediction_horizon, input_window, training_logger, args):
    """Run a single experiment configuration."""
    experiment_config = {
        'patient': patient,
        'patient_index': patient_index,
        'feature_number': feature_number,
        'prediction_horizon': prediction_horizon,
        'input_window': input_window,
        'prediction_horizon_minutes': prediction_horizon * 5,
        'input_window_minutes': input_window * 5
    }
    
    training_logger.logger.info(
        f"Starting experiment: {patient} | "
        f"Features: {feature_number} | "
        f"Input: {input_window*5}min | "
        f"Prediction: {prediction_horizon*5}min"
    )
    
    if args.dry_run:
        # Return dummy results for dry run
        return {
            'rmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'r2': 0.0,
            'y_true': [], 'y_pred': [], 'model': None
        }
    
    # Run the actual experiment
    rmse, mae, mape, r2, y_true, y_pred, model = test_model(
        testFlag=1,  # Always test
        patientFlag=patient_index,
        layerNumber=config['model']['num_layers'],
        plotFlag=0,  # Disable plotting in batch mode
        featureNumber=feature_number,
        horizon=prediction_horizon,
        input_window=input_window
    )
    
    # Log results
    results = {
        'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2
    }
    
    training_logger.logger.info(
        f"Results - RMSE: {rmse:.4f}, MAE: {mae:.4f}, "
        f"MAPE: {mape:.4f}%, R²: {r2:.4f}"
    )
    
    return {
        **results, 
        'y_true': y_true, 
        'y_pred': y_pred, 
        'model': model,
        'config': experiment_config
    }

def save_experiment_data(config, patient, results, output_dir):
    """Save experiment data for important patients."""
    important_patients = ['patient575', 'patient552']
    
    if patient in important_patients and config['logging']['save_model']:
        # Save model
        if results['model'] is not None:
            model_path = os.path.join(output_dir, f"{patient}_model.pth")
            torch.save(results['model'].state_dict(), model_path)
        
        # Save predictions and actual values
        if config['logging']['save_predictions']:
            actual_df = pd.DataFrame({
                'index': range(len(results['y_true'].flatten())),
                'glucose': results['y_true'].flatten()
            })
            actual_path = os.path.join(output_dir, f"{patient}_actual_values.csv")
            actual_df.to_csv(actual_path, index=False, header=False)
            
            pred_df = pd.DataFrame({
                'index': range(len(results['y_pred'].flatten())),
                'glucose': results['y_pred'].flatten()
            })
            pred_path = os.path.join(output_dir, f"{patient}_predicted_values.csv")
            pred_df.to_csv(pred_path, index=False, header=False)

def run_experiments(config, args):
    """Run all experiment configurations."""
    start_time = time.time()
    
    # Setup environment
    output_dir, log_dir = setup_environment(config, args)
    
    # Setup logging
    logger = setup_logging(config)
    writer = setup_tensorboard(config)
    training_logger = TrainingLogger(config, writer)
    
    training_logger.start_training()
    
    # Get experiment parameters
    patients = config['data']['patients']
    input_windows = config['experiment_params']['input_windows']
    prediction_horizons = config['experiment_params']['prediction_horizons']
    feature_numbers = config['experiment_params']['feature_numbers']
    
    all_results = {}
    
    try:
        for prediction_horizon in prediction_horizons:
            for input_window in input_windows:
                # Create Excel workbook for this configuration
                workbook_name = os.path.join(
                    output_dir, 
                    f"glucose_prediction_results_pred{prediction_horizon*5}min_input{input_window*5}min.xlsx"
                )
                workbook = xlsxwriter.Workbook(workbook_name)
                
                config_results = {
                    'prediction_horizon': prediction_horizon,
                    'input_window': input_window,
                    'results_by_features': {}
                }
                
                for feature_number in feature_numbers:
                    worksheet_name = f"Features_{feature_number}"
                    worksheet = workbook.add_worksheet(worksheet_name)
                    
                    # Setup worksheet headers
                    headers = ['Patient', 'Test RMSE', 'Test MAE', 'Test MAPE', 'Test R^2']
                    for col, header in enumerate(headers):
                        worksheet.write(0, col, header)
                    
                    feature_results = {
                        'feature_number': feature_number,
                        'patient_results': {}
                    }
                    
                    metrics_lists = {'rmse': [], 'mae': [], 'mape': [], 'r2': []}
                    
                    for patient_index, patient in enumerate(patients):
                        # Run experiment
                        experiment_result = run_single_experiment(
                            config, patient_index, patient, feature_number,
                            prediction_horizon, input_window, training_logger, args
                        )
                        
                        # Store results
                        feature_results['patient_results'][patient] = experiment_result
                        
                        # Collect metrics
                        metrics_lists['rmse'].append(experiment_result['rmse'])
                        metrics_lists['mae'].append(experiment_result['mae'])
                        metrics_lists['mape'].append(experiment_result['mape'])
                        metrics_lists['r2'].append(experiment_result['r2'])
                        
                        # Write to Excel
                        row = patient_index + 1
                        worksheet.write(row, 0, patient)
                        worksheet.write(row, 1, experiment_result['rmse'])
                        worksheet.write(row, 2, experiment_result['mae'])
                        worksheet.write(row, 3, experiment_result['mape'])
                        worksheet.write(row, 4, experiment_result['r2'])
                        
                        # Save experiment data
                        save_experiment_data(config, patient, experiment_result, output_dir)
                    
                    # Calculate and log summary statistics
                    summary_stats = {
                        f'mean_{metric}': sum(values) / len(values)
                        for metric, values in metrics_lists.items()
                    }
                    feature_results['summary'] = summary_stats
                    
                    training_logger.logger.info(
                        f"Summary for {feature_number} features: "
                        f"RMSE: {summary_stats['mean_rmse']:.4f}, "
                        f"MAE: {summary_stats['mean_mae']:.4f}, "
                        f"MAPE: {summary_stats['mean_mape']:.4f}%, "
                        f"R²: {summary_stats['mean_r2']:.4f}"
                    )
                    
                    config_results['results_by_features'][feature_number] = feature_results
                
                workbook.close()
                training_logger.logger.info(f"Results saved to {workbook_name}")
                
                all_results[f"pred{prediction_horizon}_input{input_window}"] = config_results
        
        # Final summary
        end_time = time.time()
        total_time = end_time - start_time
        
        final_results = {
            'total_execution_time_minutes': total_time / 60,
            'experiment_results': all_results,
            'configuration': config
        }
        
        training_logger.log_final_results({
            'Total_Time_Minutes': total_time / 60,
            'Experiments_Completed': len(all_results)
        })
        
        # Save complete experiment results
        save_paths = save_experiment_results(
            config, final_results, 
            model_state=None  # Models saved individually
        )
        
        training_logger.logger.info(f"Complete results saved to: {save_paths['results_path']}")
        
    except Exception as e:
        training_logger.logger.error(f"Experiment failed with error: {str(e)}")
        raise
    finally:
        training_logger.close()
    
    return final_results

def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Configuration file {args.config} not found!")
        sys.exit(1)
    
    # Override config with command line arguments
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
    
    print(f"Starting TW-MRT experiment: {config['experiment']['name']}")
    print(f"Configuration loaded from: {args.config}")
    
    if args.dry_run:
        print("DRY RUN MODE - No actual training will be performed")
    
    # Run experiments
    results = run_experiments(config, args)
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Total time: {results['total_execution_time_minutes']:.2f} minutes")
    print(f"Results saved in: {config['experiment']['output_dir']}")
    print(f"Logs saved in: {config['experiment']['log_dir']}")

if __name__ == "__main__":
    main() 