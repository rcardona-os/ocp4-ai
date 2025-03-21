import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import time

# Read environment config
DATA_PATH = os.getenv("DATA_PATH", "/workspace/WMT_combined.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/model.pth")
PLOT_PATH = os.getenv("PLOT_PATH", "/workspace/prediction_plot.png")
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"

# Device setup
device = torch.device("cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"🚀 Using device: {device}")

# Load dataset
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df["Next_Close"] = df["Close"].shift(-1)
df.dropna(inplace=True)

features = ["Open", "High", "Low", "Close", "Volume"]
X = torch.tensor(df[features].values, dtype=torch.float32)
y = torch.tensor(df["Next_Close"].values, dtype=torch.float32).unsqueeze(1)

# Split dataset
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# DataLoader
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# Define model
model = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
).to(device)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Train
epochs = 20
losses = []
start = time.time()
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

# Predict and save plot
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy()
    y_true = y_test.numpy()

plt.figure(figsize=(12, 5))
plt.plot(y_true[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted", linestyle="--")
plt.title("Next-Day Close Price Prediction (PyTorch)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"📊 Plot saved to {PLOT_PATH}")