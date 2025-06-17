import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 =False
    torch.backends.cuda.matmul.allow_tf32=False
set_seed(0)
def dataProcess(featureNumber, patientFlag, horizon, input_window):
    set_seed(0)
    patientTrainList = [ 'all_data/552training.csv',
                         'all_data/575training.csv']

    patientTestList = ['all_data/552testing.csv',
                       'all_data/575testing.csv']

    dataset = pd.read_csv(patientTrainList[patientFlag], header=0, index_col=0, usecols=[i for i in range(featureNumber + 1)])
    test_dataset = pd.read_csv(patientTestList[patientFlag], header=0, index_col=0, usecols=[i for i in range(featureNumber + 1)])

    for df in [dataset, test_dataset]:
        df['CGM'] = df['CGM'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        if 'CHO' in df.columns:
            df['CHO'] = df['CHO'].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
    normal_data = dataset['CGM'].copy()
    def fx(x, dt):
        return np.array([x[0]])

    def hx(x):
        return np.array([x[0]])

    def apply_ukf_to_series(series):
        points = MerweScaledSigmaPoints(n=1, alpha=0.001, beta=2.0, kappa=0)
        ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1, dt=1, fx=fx, hx=hx, points=points)
        ukf.x = np.array([series.iloc[0]])
        ukf.P *= 10.
        ukf.R = 0.1
        ukf.Q = 0.1

        smoothed_series = []
        for measurement in series:
            ukf.predict()
            ukf.update(np.array([measurement]))
            smoothed_series.append(ukf.x[0])

        return smoothed_series
    smoothed_data = apply_ukf_to_series(dataset['CGM'])
    dataset['CGM'] = smoothed_data
    test_dataset['CGM'] = apply_ukf_to_series(test_dataset['CGM'])
    import matplotlib.pyplot as plt
    n = 300
    normal_data = normal_data[:n]
    smoothed_data = smoothed_data[:n]

    # Create index array
    index = np.arange(n)
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot normal data with a solid line
    ax.plot(index, normal_data, color='#FF4136', linewidth=2, label='Gerçek Veri')

    # Plot smoothed data with a dashed line
    ax.plot(index, smoothed_data, color='#0074D9', linewidth=2, linestyle='--', label='Yumuşatılmış Veri (UKS)')

    # Set x-axis ticks and labels
    ax.set_xticks(np.arange(0, n+1, 50))
    ax.set_xticklabels(np.arange(0, n+1, 50))

    # Set labels and title
    ax.set_xlabel('Zaman İndeksi', fontsize=12)
    ax.set_ylabel('Glikoz Seviyesi (mg/dL)', fontsize=12)
    ax.set_title(f'Gerçek ve Yumuşatılmış Glikoz Verisi - {"Hasta 552"}')

    # Add legend
    ax.legend(fontsize=10)

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.show()
    data_dict = {
    'Index': index,
    'Normal Data': normal_data,
    'Smoothed Data': smoothed_data
}
    df = pd.DataFrame(data_dict)

    # Save to Excel
    file_path = 'plot_data.xlsx'
    df.to_excel(file_path, index=False)
    #plt.close(fig)
    #plt.close()
    dataset_train = dataset.to_numpy().reshape(-1, featureNumber)
    test_dataset_test = test_dataset.to_numpy().reshape(-1, featureNumber)

    scaler = StandardScaler()
    dataset_train = scaler.fit_transform(dataset_train)
    test_dataset_test = scaler.transform(test_dataset_test)

    train_data, val_data = train_test_split(dataset_train, test_size=0.20, shuffle=False)

    def to_sequences(seq_size, output_size, obs, target_index=0):
        x, y = [], []
        for i in range(len(obs) - seq_size - output_size + 1):
            x.append(obs[i:(i + seq_size), :])
            y.append(obs[i + seq_size:i + seq_size + output_size, target_index])
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).view(-1, output_size))

    x_train, y_train = to_sequences(input_window, horizon, train_data)
    x_val, y_val = to_sequences(input_window, horizon, val_data)
    x_test, y_test = to_sequences(input_window, horizon, test_dataset_test)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader, scaler, y_test, normal_data, smoothed_data