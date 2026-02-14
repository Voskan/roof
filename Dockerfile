# Use NVIDIA PyTorch image as base for CUDA support and pre-installed libraries
# nvcr.io/nvidia/pytorch:24.01-py3 contains PyTorch 2.1.2 + CUDA 12.3
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set working directory
WORKDIR /workspace/DeepRoof-2026

# Install system dependencies
# libgl1-mesa-glx is required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Common OpenMMLab dependencies
# MIM is the OpenMMLab command line interface for installing OpenMMLab packages
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.0"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to keep image size small
RUN pip install --no-cache-dir -r requirements.txt

# Install MMDetection and MMSegmentation
# These are often better installed via MIM or after other dependencies
RUN mim install "mmdet>=3.0.0"
RUN mim install "mmsegmentation>=1.0.0"

# Expose port (if needed for API/Jupyter)
EXPOSE 8888

# Default command
CMD ["/bin/bash"]
