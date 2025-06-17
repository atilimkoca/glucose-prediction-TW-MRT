import os
import xlsxwriter
import time
from testModel import test_model
from testdataprocess import dataProcess
from plot import create_surveillance_error_grid, plot_smoothed_vs_normal
# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Constants
SEED_RANGE = 1
MAX_EPOCHS = 200
PATIENT_NUMBER = 12
LAYER_NUMBER = 3
TEST_FLAG = 1
PLOT_FLAG = 0

# Patient list
PATIENTS = ['patient552', 'patient575']

# Dynamic parameters
INPUT_WINDOWS = [3]  # 15, 30, 45, 60, 90, 120 minutes (in 5-minute intervals)
PREDICTION_HORIZONS = [12]  # 15, 30, 45, 60, 90, 120 minutes (in 5-minute intervals)
FEATURE_NUMBERS = [3]  # CGM, CGM+Basal, CGM+Basal+CHO, CGM+Basal+CHO+Bolus

import os
import pandas as pd
import torch
import time
from testModel import test_model

def run_experiments():
    start_time = time.time()
    
    # Create a directory to store results if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    for prediction_horizon in PREDICTION_HORIZONS:
        for input_window in INPUT_WINDOWS:
            workbook_name = f"results/new_with_best_hyper_glucose_prediction_results_pred{prediction_horizon*5}min_input{input_window*5}min.xlsx"
            workbook = xlsxwriter.Workbook(workbook_name)
            
            for feature_number in FEATURE_NUMBERS:
                worksheet_name = f"Features_{feature_number}"
                worksheet = workbook.add_worksheet(worksheet_name)
                worksheet.write('A1', 'Patient')
                worksheet.write('B1', 'Test RMSE')
                worksheet.write('C1', 'Test MAE')
                worksheet.write('D1', 'Test MAPE')
                worksheet.write('E1', 'Test R^2')
                
                rmse_test_list = []
                mae_test_list = []
                mape_test_list = []
                r2_test_list = []
                
                for patient_index, patient in enumerate(PATIENTS):
                    print(f"Processing {patient} with {feature_number} features, "
                          f"{input_window*5}min input, {prediction_horizon*5}min prediction")
                    
                    rmse_test, mae_test, mape_test, r2_test, y_true, y_pred, model = test_model(
                        TEST_FLAG, patient_index, LAYER_NUMBER, PLOT_FLAG, 
                        feature_number, prediction_horizon, input_window
                    )
                    rmse_test_list.append(rmse_test)
                    mae_test_list.append(mae_test)
                    mape_test_list.append(mape_test)
                    r2_test_list.append(r2_test)
                    
                    worksheet.write_column(1, 0, PATIENTS)
                    worksheet.write_column(1, 1, rmse_test_list)
                    worksheet.write_column(1, 2, mae_test_list)
                    worksheet.write_column(1, 3, mape_test_list)
                    worksheet.write_column(1, 4, r2_test_list)
                    
                    # Save model and data for patients 575 and 552
                    if patient in ['patient575', 'patient552']:
                        # Save model
                        torch.save(model.state_dict(), f"results/{patient}_model.pth")
                        
                        # Save actual values to CSV
                        df_actual = pd.DataFrame({
                            'index': range(len(y_true.flatten())),
                            'glucose': y_true.flatten()
                        })
                        df_actual.to_csv(f"results/{patient}_actual_values.csv", index=False, header=False)
                        
                        # Save predicted values to CSV
                        df_pred = pd.DataFrame({
                            'index': range(len(y_pred.flatten())),
                            'glucose': y_pred.flatten()
                        })
                        df_pred.to_csv(f"results/{patient}_predicted_values.csv", index=False, header=False)
                        
                        print(f"Model and data saved for {patient}")
            
            workbook.close()
            print(f"Results saved in {workbook_name}")
    
    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    run_experiments()