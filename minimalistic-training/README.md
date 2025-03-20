#### - provision container image

- creating Dockerfile file
```bash
$ mkdir image
```

```bash
$ cat image/Dockerfile
```

```text
FROM pytorch/pytorch:latest

# Install Jupyter
RUN pip install jupyterlab

# Create workspace directory
WORKDIR /workspace
COPY train.py /workspace/train.py

# Expose Jupyter Notebook port
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
```

- buildin the container image
```bash
$ podman build -t quay.io/rcardona/k8s-simple-train:latest image/
```

```bash
$ podman push quay.io/k8s-simple-train:latest
```


#### - creare namespace
```bash
$ oc new-project datascience
```

