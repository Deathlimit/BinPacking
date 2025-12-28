import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os


class MLPPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(), 
        )

    def forward(self, x):
        return self.net(x)


def train(dataset_path="dataset_binpacking.npz", epochs=50, batch_size=2048, lr=0.0005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    data = np.load(dataset_path)
    X = torch.from_numpy(data["observations"]).float()
    y = torch.from_numpy(data["actions"]).float()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLPPolicy(input_dim=X.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            
            total_loss += float(loss.item()) * xb.size(0)
            
        avg = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} - MSE: {avg:.6f}")


    torch.save(model.cpu().state_dict(), "PackingModelLast.pth")
    print("Saved model: PackingModelLast.pth")

    return model


if __name__ == "__main__":
    train()