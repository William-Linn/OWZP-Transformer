#%%
import matplotlib.pyplot as plt
import optuna
import random
import joblib
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import shap


#%%
# Load data
path = '/Users/lzh/Desktop/'

with open(path+'train_dataset.pkl', 'rb') as f:
    X_train, y_train = pickle.load(f)

with open(path+'val_dataset.pkl', 'rb') as f:
    X_val, y_val = pickle.load(f)

with open(path+'test_dataset.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

# Load the scaler for the test set
scaler_y_test = joblib.load(path+'y_test_scaler.pkl')

# Convert to PyTorch dataset
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

# Create DataLoader
batch_size = 128  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print the size of the datasets
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

#%%
# Train
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np

# Define DataEmbedding layer
class DataEmbedding(nn.Module):
    def __init__(self, d_model, dropout):
        super(DataEmbedding, self).__init__()
        self.linear = nn.Linear(15, d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        return self.dropout(x)

# Define PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, nin, nout, d_model, nhead, nhid, num_encoder_layers, num_decoder_layers, dropout=0.25):
        super(TransformerModel, self).__init__()
        self.embedding = DataEmbedding(d_model, dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Define the first encoder
        encoder_layer_1 = nn.TransformerEncoderLayer(d_model, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_encoder_layers)

        # Define the second encoder
        encoder_layer_2 = nn.TransformerEncoderLayer(d_model, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_encoder_layers)

        # Define the decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, nhid, dropout, activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)

        # Define the output layer
        self.output_layer = nn.Linear(d_model, nout * 5) 

        self.d_model = d_model
        self.nin = nin
        self.nout = nout
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.linear.weight.data.uniform_(-initrange, initrange)
        self.embedding.linear.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()

    def forward(self, src, tgt=None, return_attention=False):
        if tgt is None:
            tgt = torch.zeros_like(src)  

        # Embedding and positional encoding for source
        src = self.embedding(src)
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        src = self.pos_encoder(src * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)))

        # Embedding and positional encoding for target
        tgt = self.embedding(tgt)
        tgt = tgt.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        tgt = self.pos_encoder(tgt * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)))

        # First encoder
        memory_1 = self.transformer_encoder_1(src)
        # Second encoder
        memory_2 = self.transformer_encoder_2(memory_1)

        # Decoder with manual attention weights extraction
        output, attn_weights = self.decoder_layer.multihead_attn(tgt, memory_2, memory_2)  
        output = self.transformer_decoder(tgt, memory_2)
        output = self.output_layer(output)  # Apply the output layer
        output = output[-1, :, :].unsqueeze(0)  # Extract the last time step [1, batch_size, output_dim]

        if return_attention:
            return output.permute(1, 0, 2), attn_weights  
        else:
            return output.permute(1, 0, 2)  # [batch_size, 1, output_dim]

#Set hyperparameters
seq_len = 5  # Number of input steps
label_len = 1  # Number of output steps
d_model = 128  # Embedding dimension
nhead = 16  # Number of heads
nhid = 128  # Dimension of the feedforward network
num_encoder_layers = 1
num_decoder_layers = 2
dropout = 0.1 # Dropout rate


# Create model instance
model = TransformerModel(seq_len, label_len, d_model, nhead, nhid, num_encoder_layers, num_decoder_layers, dropout)


# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

# Early stopping setting
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), '/Users/lzh/Desktop/TF_checkpoint.pt')



# MultiTaskLoss
# --------------------------
# Haversine loss for track (latitude, longitude) in kilometers
def haversine_loss(lat_pred, lon_pred, lat_true, lon_true):
    R = 6371.0  # Earth radius in km
    phi1 = torch.deg2rad(lat_true)
    phi2 = torch.deg2rad(lat_pred)
    dphi = phi2 - phi1
    dlambda = torch.deg2rad(lon_pred - lon_true)
    a = torch.sin(dphi/2)**2 + torch.cos(phi1)*torch.cos(phi2)*torch.sin(dlambda/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    return (R * c).mean()

    

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, huber_delta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        # PyTorch >=1.9
        # self.huber = getattr(nn, 'HuberLoss', nn.SmoothL1Loss)(delta=huber_delta)
        self.inten_loss = nn.MSELoss()  

    def forward(self, preds, targets):
        # ---- 1. flatten preds/targets to shape [batch, 5] ----
        # if shape is [B,1,5] -> squeeze dim1
        if preds.dim() == 3 and preds.size(1) == 1:
            preds = preds.squeeze(1)
        # if shape is [B,5,1] or similar, you may need .view
        if preds.dim() != 2 or preds.size(1) != 5:
            preds = preds.view(preds.size(0), -1)
        # process targets in the same way
        if targets.dim() == 3 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        if targets.dim() != 2 or targets.size(1) != 5:
            targets = targets.view(targets.size(0), -1)

        # ---- 2. Track loss: Haversine on lat/lon ----
        lat_p, lon_p = preds[:, 0], preds[:, 1]
        lat_t, lon_t = targets[:, 0], targets[:, 1]
        loss_track = haversine_loss(lat_p, lon_p, lat_t, lon_t)

        # ---- 3. Intensity loss: Huber on last three columns ----
        # loss_inten = self.huber(preds[:, 2:], targets[:, 2:])
        # intensity
        mse = self.inten_loss(preds[:,2:], targets[:,2:])
        loss_inten = torch.sqrt(mse + 1e-6)  # add eps to ensure numerical stability
        
        # ---- 4. Combine ----
        return self.alpha * loss_track + self.beta * loss_inten




# Model training function
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.0005):
    # criterion = MultiTaskLossHomoUnc()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    early_stopping = EarlyStopping(patience=20, delta=0.001)
    criterion = MultiTaskLoss(alpha=1.0, beta=1.0)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            # Define target tensor
            tgt = torch.zeros_like(inputs)

            outputs = model(inputs, tgt, return_attention=False)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                tgt = torch.zeros_like(inputs)
                outputs = model(inputs, tgt)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # # Plot training loss and validation loss
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    # plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.show()

# Train the model
train_model(model, train_loader, val_loader)

# Load the best model for testing
model.load_state_dict(torch.load('/Users/lzh/Desktop/TF_checkpoint.pt'))
model.eval()


# Test the model and calculate RMSE
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        tgt = torch.zeros_like(inputs)
        outputs = model(inputs, tgt)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

test_loss /= len(test_loader.dataset)
print(f'Test Loss (RMSE): {test_loss:.4f}')


#%%
# Test
import torch
import numpy as np
import joblib
from torch.utils.data import DataLoader, TensorDataset

path = '/Users/lzh/Desktop/'
# Load data from file
with open(path+'test_dataset.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

# Load scaler for the test set
scaler_y_test = joblib.load(path+'y_test_scaler.pkl')

# Convert to PyTorch dataset
batch_size = 128
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def haversine_km(lon1, lat1, lon2, lat2):
    """
    all args: 1-D ndarray or list, degrees
    return   : 1-D ndarray, great-circle distance (km)
    """
    R = 6371.0  # Earth's mean radius (km)
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )
    dlon  = lon2 - lon1
    dlat  = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c            # km

# Inverse standardization function
def inverse_transform(y, scaler):
    y = y.reshape(-1, y.shape[-1])  # reshape to (N, 5)
    y = scaler.inverse_transform(y)
    return y

# Define metrics calculation function
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    ss_res = np.sum((y_pred - y_true) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1 - (ss_res / ss_tot)
    bias  = np.mean(y_pred - y_true, axis=0)    
    return rmse, mae, r2, bias

# Test the model
model.load_state_dict(torch.load('/Users/lzh/Desktop/TF_checkpoint_F2.pt'))
model.eval()
test_loss = 0.0
all_preds = []
all_trues = []

criterion = MultiTaskLoss(alpha=1.0, beta=1.0)

with torch.no_grad():
    for inputs, labels in test_loader:
        tgt = torch.zeros_like(inputs)
        outputs = model(inputs, tgt)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

        all_preds.append(outputs.cpu().numpy())
        all_trues.append(labels.cpu().numpy())

# Average test loss
test_loss /= len(test_loader.dataset)
print(f'Test Loss (RMSE): {test_loss:.4f}')

# Merge predictions and ground truths from all batches
all_preds = np.concatenate(all_preds, axis=0)
all_trues = np.concatenate(all_trues, axis=0)

# Perform inverse scaling for predictions and ground truths at once
all_preds_inversed = inverse_transform(all_preds, scaler_y_test)
all_trues_inversed = inverse_transform(all_trues, scaler_y_test)

# Calculate metrics for each variable
rmse_all, mae_all, r2_all, bias_all = calculate_metrics(all_trues_inversed, all_preds_inversed)

variables = ['Latitude', 'Longitude', 'Max Wind Speed','Pressure','Future 24h Intensity Change']

# # Print metrics for each variable
# for i, var in enumerate(variables):
#     print(f'{var}:')
#     print('RMSE:', rmse_all[i])
#     print('MAE:', mae_all[i])
#     print('R^2:', r2_all[i])
#     print('Bias :', bias_all[i])
#     print(" ")

# B. Compute Track-level Haversine error (km)
# ------------------------------------------------------------
track_dist_km = haversine_km(
    all_trues_inversed[:, 1],   # lon_true
    all_trues_inversed[:, 0],   # lat_true
    all_preds_inversed[:, 1],   # lon_pred
    all_preds_inversed[:, 0],   # lat_pred
)
track_rmse_km = np.sqrt(np.mean(track_dist_km**2))
track_mae_km  = np.mean(np.abs(track_dist_km))
track_bias_km = np.mean(track_dist_km)      # positive = average "too far"

# ------------------------------------------------------------
# C. Organize metrics for 6 variables and print
# ------------------------------------------------------------
variables = ['Track', 'Latitude', 'Longitude',
             'Max Wind Speed', 'Pressure',
             'Future 24h Intensity Change']

rmse_show  = np.concatenate(([track_rmse_km],  rmse_all))
mae_show   = np.concatenate(([track_mae_km],   mae_all))
bias_show  = np.concatenate(([track_bias_km],  bias_all))
# No R² for Track, assign NaN
r2_show    = np.concatenate(([np.nan],         r2_all))

for i, var in enumerate(variables):
    print(f'{var}:')
    print('  RMSE:', rmse_show[i])
    print('  MAE :', mae_show[i])
    if not np.isnan(r2_show[i]):
        print('  R^2 :', r2_show[i])
    print('  Bias:', bias_show[i])
    print()

# ------------------------------------------------------------
# Future-24 h Intensity Change : scatter (true vs pred)
# ------------------------------------------------------------
import matplotlib.pyplot as plt

col_idx = variables.index('Future 24h Intensity Change') - 1  # Track occupies column 0
y_true_fc = all_trues_inversed[:, col_idx]
y_pred_fc = all_preds_inversed[:, col_idx]

rmse_fc = rmse_all[col_idx]
mae_fc  = mae_all[col_idx]
r2_fc   = r2_all[col_idx]
bias_fc = bias_all[col_idx]

plt.figure(figsize=(6, 6))
plt.scatter(y_true_fc, y_pred_fc, s=18, alpha=0.6)

min_val = min(y_true_fc.min(), y_pred_fc.min())
max_val = max(y_true_fc.max(), y_pred_fc.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)

plt.xlabel('True ΔV$_{24h}$ (m s$^{-1}$)')
plt.ylabel('Predicted ΔV$_{24h}$ (m s$^{-1}$)')
plt.title('Future 24 h Intensity Change')

# Write four metrics into the legend
stats = (f'RMSE={rmse_fc:.2f}\n'
          f'MAE ={mae_fc:.2f}\n'
          f'R$^2$ ={r2_fc:.3f}\n'
          f'Bias={bias_fc:.2f}')
plt.legend([stats], loc='upper left', frameon=True)

plt.grid(ls=':')
plt.tight_layout()
plt.savefig('/Users/lzh/Desktop/uture24_scatter.png', dpi=300)
plt.show()




