import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import time

# File paths
DATA_PATH = os.getenv("DATA_PATH", "/workspace/WMT.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/model.pth")
PLOT_PATH = os.getenv("PLOT_PATH", "/workspace/prediction_plot.png")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Load data
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df["Next_Close"] = df["Close"].shift(-1)
df.dropna(inplace=True)

features = ["Open", "High", "Low", "Close", "Volume"]
X = torch.tensor(df[features].values, dtype=torch.float32)
y = torch.tensor(df["Next_Close"].values, dtype=torch.float32).unsqueeze(1)

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# Define model
model = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
).to(device)

# Loss and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
start = time.time()
epochs = 20
losses = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
end = time.time()

print(f"⏱️ Training time: {end - start:.2f} seconds")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")

# Evaluate and plot
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy()
    y_true = y_test.numpy()

plt.figure(figsize=(12, 5))
plt.plot(y_true[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted", linestyle="--")
plt.title("PyTorch: Next-Day Close Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"📊 Prediction plot saved to {PLOT_PATH}")