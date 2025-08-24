# Step 1: Start with an official, lightweight Python base image.
FROM python:3.11-slim

# Step 2: Set a working directory inside the container.
WORKDIR /app

# Step 3: Install essential system libraries.
# This is a crucial step that pip can't do. It installs tools
# that libraries like OpenCV and TensorFlow need to build correctly on Linux.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements file into the container first.
# This allows Docker to cache the installed packages unless this file changes.
COPY requirements.txt .

# Step 5: Install all Python dependencies using the robust requirements list.
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your application code into the container.
COPY . .

# Step 7: Tell Docker what command to run when the container starts.
# This is the new, correct, dynamic line
CMD ["gunicorn", "run:app", "--bind", "0.0.0.0:$PORT"]