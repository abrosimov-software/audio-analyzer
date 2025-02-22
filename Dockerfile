# Use an official PyTorch image with CUDA support
# FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install essential OS packages
# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     git \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt before installing dependencies
COPY . .

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

