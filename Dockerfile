# Use a lightweight Python image
FROM python:3.11-slim

# Install system dependencies including tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy dependencies first and install them
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the code
COPY . .

# Run the app with Gunicorn on Render's required port
CMD ["gunicorn", "-b", "0.0.0.0:$PORT", "app:app"]
