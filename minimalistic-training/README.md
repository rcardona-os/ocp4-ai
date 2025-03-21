#### 0 - Prerequisites

- [x] Openshift v4.18
- [x] Default storage class block based
- [x] Nvidia GPU capable worker

#### 1 - Creating the model

```bash
$ cat train_torch.py
```

```text
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
```

#### 2 - (OPTIONAL) Create the container image with model application

```bash
$ cat Dockerfile
```

```text
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Install Python packages
RUN pip install pandas matplotlib

# Copy files
COPY train_torch.py .
COPY WMT_combined.csv .

# Run training
CMD ["python", "train_torch.py"]
```

- building the container image 
```bash
$ podman build -t quay.io/rcardona/walmart-train-demo:latest .
```

```bash
$ podman push quay.io/rcardona/walmart-train-demo:latest
```

#### 3 - Create namespace
```bash
$ oc new-project stock-walmart
```

#### 4 - Persiting model & plot
```bash
$ cat << EOF | oc apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
EOF
```


#### 4 - Training with CPUs
```bash
$ cat << EOF | oc apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: train-cpu-job
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: your-registry/walmart-train-demo:latest
        env:
        - name: FORCE_CPU
          value: "true"
        - name: DATA_PATH
          value: "/workspace/WMT_combined.csv"
        - name: MODEL_PATH
          value: "/workspace/model.pth"
        - name: PLOT_PATH
          value: "/workspace/prediction_plot.png"
        volumeMounts:
        - name: model-storage
          mountPath: /workspace
        resources:
          limits:
            cpu: "1"
            memory: "2Gi"
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage
EOF
```

#### 5 - Training with CPUs
```bash
$ cat << EOF | oc apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: train-gpu-job
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: your-registry/walmart-train-demo:latest
        env:
        - name: DATA_PATH
          value: "/workspace/WMT_combined.csv"
        - name: MODEL_PATH
          value: "/workspace/model.pth"
        - name: PLOT_PATH
          value: "/workspace/prediction_plot.png"
        volumeMounts:
        - name: model-storage
          mountPath: /workspace
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "4Gi"
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage
EOF
```

#### - Check the model
```bash
$ oc logs -f 
```