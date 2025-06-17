import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score
from testdataprocess import dataProcess
import logging

logging.basicConfig(level=logging.INFO)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DynamicFeatureImportanceLayer(nn.Module):
    def __init__(self, max_features):
        super().__init__()
        self.importance = nn.Parameter(torch.ones(max_features))
    
    def forward(self, x):
        return x * self.importance[:x.size(2)].unsqueeze(0).unsqueeze(0)

class TimeAwareEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, max_len=5000):
        super().__init__()
        self.time_embed = nn.Embedding(max_len, d_model)
        self.proj = nn.Linear(input_dim, d_model)
    
    def forward(self, x, times):
        return self.proj(x) + self.time_embed(times)

class GlucoseTrendAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trend_proj = nn.Linear(d_model, 3)
    
    def forward(self, x):
        trends = self.trend_proj(x.mean(dim=1))
        trend_weights = nn.functional.softmax(trends, dim=-1).unsqueeze(1).unsqueeze(1)
        return (x.unsqueeze(-1) * trend_weights).sum(dim=-1)

class MultiResolutionFusion(nn.Module):
    def __init__(self, d_model, num_resolutions=3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=2**i, padding=2**i-1) 
            for i in range(num_resolutions)
        ])
        self.fusion = nn.Linear(d_model * num_resolutions, d_model)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Transpose to [batch_size, d_model, seq_len] for Conv1d
        x = x.transpose(1, 2)
        multi_scale = [nn.functional.adaptive_avg_pool1d(conv(x), 1).squeeze(-1) for conv in self.convs]
        return self.fusion(torch.cat(multi_scale, dim=1))

class DynamicGlucosePredictionModel(nn.Module):
    def __init__(self, input_features, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1):
        super().__init__()
        self.feature_importance = DynamicFeatureImportanceLayer(input_features)
        self.time_embed = TimeAwareEmbedding(input_features, d_model)
        self.trend_attention = GlucoseTrendAttention(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.multi_res_fusion = MultiResolutionFusion(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, src, times):
        src = self.feature_importance(src)
        src = self.time_embed(src, times)
        src = self.trend_attention(src * (src.size(-1) ** 0.5))
        output = self.transformer_encoder(src.transpose(0, 1))
        output = output.transpose(0, 1)  # Change back to [batch_size, seq_len, d_model]
        return self.output_proj(self.multi_res_fusion(output))
def print_notation_summary(model, batch_size=32, input_window=300):
    device = next(model.parameters()).device
    featureNumber = model.feature_importance.importance.shape[0]
    horizon = model.output_proj.out_features

    # Input tensors
    X = torch.randn(batch_size, input_window, featureNumber).to(device)
    T = torch.arange(input_window).unsqueeze(0).repeat(batch_size, 1).to(device)
    Y = torch.randn(batch_size, horizon).to(device)

    print("\nNotation Summary:")
    print(f"\\mathbf{{X}}: Input tensor, shape {X.shape}")
    print(f"\\mathbf{{T}}: Time indices tensor, shape {T.shape}")
    print(f"\\mathbf{{Y}}: Target tensor, shape {Y.shape}")

    # Feature Importance
    F = model.feature_importance(X)
    print(f"\\mathbf{{F}}: Feature importance output, shape {F.shape}")
    print(f"\\mathbf{{W}}_{{imp}}: Feature importance weights, shape {model.feature_importance.importance.shape}")

    # Time Embedding
    E = model.time_embed(F, T)
    print(f"\\mathbf{{E}}: Time-aware embedding output, shape {E.shape}")
    print(f"\\mathbf{{W}}_{{time}}: Time embedding weights, shape {model.time_embed.time_embed.weight.shape}")
    print(f"\\mathbf{{W}}_{{proj}}: Time embedding projection, in_features={model.time_embed.proj.in_features}, out_features={model.time_embed.proj.out_features}")

    # Trend Attention
    E_w = model.trend_attention(E)
    print(f"\\mathbf{{E}}_{{w}}: Trend attention output, shape {E_w.shape}")
    print(f"\\mathbf{{W}}_{{trend}}: Trend projection, in_features={model.trend_attention.trend_proj.in_features}, out_features={model.trend_attention.trend_proj.out_features}")

    # Transformer
    E_w_transposed = E_w.transpose(0, 1)
    print(f"\\mathbf{{E}}_{{w}} (transposed): Transformer input, shape {E_w_transposed.shape}")
    
    # Multi-Resolution Fusion
    transformer_output = model.transformer_encoder(E_w_transposed).transpose(0, 1)
    O = model.multi_res_fusion(transformer_output)
    print(f"\\mathbf{{O}}: Multi-resolution fusion output, shape {O.shape}")
    print(f"\\mathbf{{W}}_{{mr}}: Fusion layer, in_features={model.multi_res_fusion.fusion.in_features}, out_features={model.multi_res_fusion.fusion.out_features}")

    # Output projection
    Y_pred = model.output_proj(O)
    print(f"\\hat{{\\mathbf{{y}}}}: Model output, shape {Y_pred.shape}")
    print(f"\\mathbf{{W}}_{{out}}: Output projection, in_features={model.output_proj.in_features}, out_features={model.output_proj.out_features}")

    # Loss
    criterion = nn.MSELoss()
    loss = criterion(Y_pred, Y)
    print(f"\\mathcal{{L}}: Loss (MSELoss), scalar")

    # Model parameters
    print(f"\\boldsymbol{{\\theta}}: All model parameters, count {sum(p.numel() for p in model.parameters())}")


def test_model(testFlag, patientFlag, layerNumber, plotFlag, featureNumber, horizon, input_window):
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data processing
    train_loader, val_loader, test_loader, scaler, y_test, normal_data, smoothed_data = dataProcess(featureNumber, patientFlag, horizon, input_window)

    # Initialize model
    model = DynamicGlucosePredictionModel(
        input_features=featureNumber,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        output_dim=horizon,
        dropout=0.1
    ).to(device)
    print_notation_summary(model, input_window=input_window)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    num_epochs = 100  # Adjust as needed
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            x_batch, y_batch = [t.to(device) for t in batch]
            times_batch = torch.arange(input_window).unsqueeze(0).repeat(x_batch.size(0), 1).to(device)
            
            optimizer.zero_grad()
            output = model(x_batch, times_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x_batch, y_batch = [t.to(device) for t in batch]
                times_batch = torch.arange(input_window).unsqueeze(0).repeat(x_batch.size(0), 1).to(device)
                output = model(x_batch, times_batch)
                loss = criterion(output, y_batch)
                val_losses.append(loss.item())

        train_loss = np.sqrt(np.mean(train_losses))
        val_loss = np.sqrt(np.mean(val_losses))
        scheduler.step(val_loss)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Testing
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = [t.to(device) for t in batch]
            times_batch = torch.arange(input_window).unsqueeze(0).repeat(x_batch.size(0), 1).to(device)
            output = model(x_batch, times_batch)
            predictions.extend(output.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Inverse transform predictions and targets
    mean_y, std_y = scaler.mean_[0], scaler.scale_[0]
    y_pred_invscaled = (predictions * std_y) + mean_y
    y_test_invscaled = (targets * std_y) + mean_y

    # Calculate metrics
    rmse = np.sqrt(np.mean((y_pred_invscaled - y_test_invscaled)**2))
    mae = np.mean(np.abs(y_pred_invscaled - y_test_invscaled))
    mape = np.mean(np.abs((y_test_invscaled - y_pred_invscaled) / y_test_invscaled)) * 100
    r2 = r2_score(y_test_invscaled, y_pred_invscaled)

    logging.info(f'RMSE: {rmse:.4f}')
    logging.info(f'MAE: {mae:.4f}')
    logging.info(f'MAPE: {mape:.4f}%')
    logging.info(f'R^2: {r2:.4f}')

    return rmse, mae, mape, r2, y_test_invscaled, y_pred_invscaled, model

# Usage
# rmse, mae, mape, r2, y_true, y_pred, model = test_model(testFlag, patientFlag, layerNumber, plotFlag, featureNumber, horizon, input_window)