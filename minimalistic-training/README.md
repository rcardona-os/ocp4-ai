#### 0 - Prerequisites

- [x] Openshift v4.18
- [x] Default storage class block based
- [x] Nvidia GPU capable worker

## Model Training

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

# Environment paths
DATA_PATH = os.getenv("DATA_PATH", "/app/WMT_combined.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/model.pth")
PLOT_PATH = os.getenv("PLOT_PATH", "/workspace/prediction_plot.png")
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"

# Device
device = torch.device("cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"🚀 Using device: {device}")

# Load CSV
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded CSV with shape: {df.shape}")
print(f"📊 Columns: {df.columns.tolist()}")

# Keep only valid columns
df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

# Preprocess
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df["Next_Close"] = df["Close"].shift(-1)
df.dropna(inplace=True)

print(f"🧼 Data shape after preprocessing: {df.shape}")

# Safety check
if df.shape[0] == 0:
    raise ValueError("📛 ERROR: Dataset is empty after preprocessing!")

# Feature selection
features = ["Open", "High", "Low", "Close", "Volume"]
X = torch.tensor(df[features].values, dtype=torch.float32)
y = torch.tensor(df["Next_Close"].values, dtype=torch.float32).unsqueeze(1)

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# DataLoader
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# Model
model = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
).to(device)

# Optimizer + loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
epochs = 20
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
    print(f"📈 Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
end = time.time()

print(f"⏱️ Training time: {end - start:.2f} seconds")

# Ensure output directories exist
import os
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")

# Predict and plot
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy()
    y_true = y_test.numpy()

plt.figure(figsize=(12, 5))
plt.plot(y_true[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted", linestyle="--")
plt.title("📊 Next-Day Close Price Prediction (PyTorch)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"📊 Plot saved to {PLOT_PATH}")
```

#### 2 - (OPTIONAL) Creating the container image with model application

```bash
$ cat Dockerfile
```

```text
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install Python packages
RUN pip install pandas matplotlib

# Copy files
COPY train_torch.py .
COPY WMT_combined.csv .

# Run training
CMD ["python", "/app/train_torch.py"]
```

- building the container image 
```bash
$ podman build -t quay.io/rcardona/walmart-train-demo:latest .
```

```bash
$ podman push quay.io/rcardona/walmart-train-demo:latest
```

#### 3 - Creating namespace
```bash
$ oc new-project stock-walmart
```

#### 4 - Persisting model & plot
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
        image: quay.io/rcardona/walmart-train-demo:latest
        env:
        - name: FORCE_CPU
          value: "true"
        - name: DATA_PATH
          value: "/app/WMT_combined.csv"
        - name: MODEL_PATH
          value: "/workspace/cpu/model.pth"
        - name: PLOT_PATH
          value: "/workspace/cpu/prediction_plot.png"
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

#### 5 - Training with GPUs
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
        image: quay.io/rcardona/walmart-train-demo:latest
        env:
        - name: DATA_PATH
          value: "/app/WMT_combined.csv"
        - name: MODEL_PATH
          value: "/workspace/gpu/model.pth"
        - name: PLOT_PATH
          value: "/workspace/gpu/prediction_plot.png"
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

----
## Model Serving
#### 6 - Model serving
```bash
$ cat serve_model.py
```

```text
import torch
from torch import nn
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

class InputData(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float

# Model setup
model = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    x = torch.tensor([[data.Open, data.High, data.Low, data.Close, data.Volume]])
    with torch.no_grad():
        prediction = model(x).item()
    return {"predicted_next_close": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

```