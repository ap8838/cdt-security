# Use official lightweight Python image
FROM python:3.13-slim

# Set work directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Default command (interactive shell)
CMD ["bash"]
