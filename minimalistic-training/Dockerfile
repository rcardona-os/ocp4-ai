FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Install Python packages
RUN pip install pandas matplotlib

# Copy files
COPY train_torch.py .
COPY WMT_combined.csv .

# Run training
CMD ["python", "train_torch.py"]