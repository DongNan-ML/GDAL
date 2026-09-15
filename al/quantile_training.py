import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import TensorDataset
from copy import deepcopy
import numpy as np
import pandas as pd
from datetime import datetime
import os

def train_model(model, train_loader, val_loader, save_dir, epochs=200, lr=0.0005, Q=100):
    """
    Train a model and keep track of the best model based on validation performance
    Returns the model with the best validation performance
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define quantiles
    quantiles = [q/Q for q in range(1, Q)]
    criterion = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
               optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_model_state = None
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_samples = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            predictions = model(X)  # [batch_size, output_size, num_quantiles]
            
            if len(y.shape) == 1:
                y = y.unsqueeze(1)

            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            train_loss += loss.item() * batch_size
            num_samples += batch_size

        # Validation
        model.eval()
        val_loss = 0
        samples_val = 0

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                predictions = model(X)
                
                if len(y.shape) == 1:
                    y = y.unsqueeze(1)

                batch_size_val = y.size(0)
                val_loss += criterion(predictions, y).item() * batch_size_val
                samples_val += batch_size_val

        avg_val_loss = val_loss / samples_val
        avg_train_loss = train_loss / num_samples

        # Record history
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch} | LR {optimizer.param_groups[0]['lr']:.4e} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = deepcopy(model.state_dict())
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')
        
        scheduler.step(avg_val_loss)

    # Load the best model before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Training completed. Loading best model with validation loss: {best_val_loss:.4f}')

    return model


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_quantiles):
        super().__init__()
        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size

        # Output layer
        self.final_layer = nn.Linear(prev_size, output_size * num_quantiles)
        self.layers = nn.Sequential(*layers)
        self.num_quantiles = num_quantiles
        self.output_size = output_size

    def forward(self, x):
        x = self.layers(x)
        x = self.final_layer(x)
        return x.reshape(-1, self.output_size, self.num_quantiles)

