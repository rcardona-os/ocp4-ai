#### - provision container image

- creating Dockerfile file
```bash
$ echo -e 'FROM pytorch/pytorch:latest

# Install Jupyter
RUN pip install jupyterlab

# Create workspace directory
WORKDIR /workspace
COPY train.py /workspace/train.py

# Expose Jupyter Notebook port
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]' | tee container-image/Dockerfile
```

- buildin the container image
```bash
$ podman build -t your-dockerhub-username/k8s-simple-train:latest .
```

```bash
$ podman push your-dockerhub-username/k8s-simple-train:latest
```


#### - creare namespace
```bash
$ oc new-project datascience
```

