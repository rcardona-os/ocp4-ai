#### - create the model

- creating a simple model algorithm
```bash
$ cat  simple-model.py
```

```text
import torch

# Create a simple model
model = torch.nn.Linear(10, 1)

# Generate dummy data
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")

print("Training complete!")
torch.save(model.state_dict(), "model.pth")
```

#### - provision container image

```bash
$ cat Dockerfile
```

```text
FROM pytorch/pytorch:latest

# Install Jupyter
RUN pip install jupyterlab

# Create workspace directory
WORKDIR /workspace
COPY simple-model.py /workspace/simple-model.py

# Expose Jupyter Notebook port
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
```

- buildin the container image
```bash
$ podman build -t quay.io/rcardona/k8s-simple-model:latest
```

```bash
$ podman push quay.io/rcardona/k8s-simple-model:latest
```


#### - creare namespace
```bash
$ oc new-project datascience
```

#### - deploy the model
```bash
apiVersion: v1
kind: Pod
metadata:
  name: training-pod
spec:
  restartPolicy: Never
  containers:
  - name: pytorch-training
    image: quay.io/rcardona/k8s-simple-model:latest
    command: ["python", "/workspace/simple-model.py"]
    resources:
      limits:
        nvidia.com/gpu: 1  # Request 1 GPU
    volumeMounts:
    - name: model-storage
      mountPath: /workspace
  volumes:
  - name: model-storage
    emptyDir: {}  # Temporary storage for model files
```