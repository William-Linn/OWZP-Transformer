#%%
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
import random

import warnings
warnings.filterwarnings('ignore')



#%%
path = '/home/zihaolin4/dl/'

with open(path+'train_dataset.pkl', 'rb') as f:
    X_train, y_train = pickle.load(f)

with open(path+'val_dataset.pkl', 'rb') as f:
    X_val, y_val = pickle.load(f)

with open(path+'test_dataset.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

scaler_y_test = joblib.load(path+'y_test_scaler.pkl')

train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

batch_size = 128  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")


#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np
import optuna

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
        torch.save(model.state_dict(), 'checkpoint.pt')

# Define RMSE loss function
def rmse_loss(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

# Model training function with Optuna
def objective(trial):
    # Define hyperparameters to be optimized
    seq_len = 6  # Number of input steps
    label_len = 1  # Number of output steps
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.001, 0.0005])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    d_model = trial.suggest_categorical('d_model', [64, 128, 256]) # Embedding Dimension
    nhead = trial.suggest_categorical('nhead', [4, 8, 16]) # Number of Heads
    nhid = trial.suggest_categorical('nhid', [64, 128, 256]) # Feedforward Network Dimension(nhid or dim_feedforward)
    num_encoder_layers = trial.suggest_categorical('num_encoder_layers', [1, 2, 3]) # Number of Encoder Layers
    num_decoder_layers = trial.suggest_categorical('num_decoder_layers', [1, 2, 3]) # Number of Decoder Layers
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4]) # Dropout Rate
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model instance
    model = TransformerModel(seq_len, label_len, d_model, nhead, nhid, num_encoder_layers, num_decoder_layers, dropout)
    
    # Use GPU if available
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    early_stopping = EarlyStopping(patience=10, delta=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            
            # Define target tensor
            tgt = torch.zeros_like(inputs).to(inputs.device)

            outputs = model(inputs, tgt)
            loss = rmse_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move to GPU if available
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                tgt = torch.zeros_like(inputs).to(inputs.device)
                outputs = model(inputs, tgt)
                loss = rmse_loss(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return val_loss

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# Print the best hyperparameters
print(f'Best hyperparameters: {study.best_params}')

