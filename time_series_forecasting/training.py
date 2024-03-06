import torch.nn as nn
import torch.optim as optim

def train_cnn_forecasting_model(model, data_loader, lr=0.001, epochs=3):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for features, targets in data_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
