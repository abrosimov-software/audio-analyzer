# Use an official PyTorch image with CUDA support
# FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the application code
COPY . .

# Configure environment variables
ENV PYTHONPATH=/app

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]

