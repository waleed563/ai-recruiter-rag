# Use official Python slim image
# slim = smaller size, no unnecessary packages
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first
# Docker caches this layer — if requirements don't change
# it won't reinstall packages on every build (saves time)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose both ports
# FastAPI runs on 8000
# Streamlit runs on 8501
EXPOSE 8000
EXPOSE 8501