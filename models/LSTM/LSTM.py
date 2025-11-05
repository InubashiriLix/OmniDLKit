from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        device: torch.device | None = None,
    ):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device if device is not None else torch.device("cpu")

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)
        
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np

    def make_linear_dataloader(
        num_samples: int = 1000,
        seq_length: int = 10,
        input_size: int = 1,
        batch_size: int = 32
    ) -> DataLoader:
        """Create a simple linear regression dataset for LSTM training."""
        X = []
        y = []
        
        for _ in range(num_samples):
            start = np.random.rand()
            sequence = np.linspace(start, start + 1, seq_length)
            X.append(sequence.reshape(-1, input_size))
            y.append(start + 1.5)
        
        X = torch.FloatTensor(np.array(X))
        y = torch.FloatTensor(np.array(y)).reshape(-1, 1)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def train_lstm():
        """Train the LSTM model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Hyperparameters
        input_size = 1
        hidden_size = 64
        output_size = 1
        num_layers = 2
        num_epochs = 50
        learning_rate = 0.001
        
        # Create dataloaders
        train_loader = make_linear_dataloader(num_samples=1000, batch_size=32)
        val_loader = make_linear_dataloader(num_samples=200, batch_size=32)
        
        # Initialize model
        model = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            device=device
        )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        print("Training completed!")
        return model
    
    # Run training
    trained_model = train_lstm()
