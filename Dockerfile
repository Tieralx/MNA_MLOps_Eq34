#Start from a lightweight Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC with default local remote support
RUN pip install dvc fastapi uvicorn joblib

# Copy the necessary project files into the container
# Adjust this to the exact structure of your Cookiecutter project
COPY dvc.yaml dvc.lock ./
ADD src ./src
# for necessary data folder if you're tracking it in Git
ADD data ./data       
# for local model files if they're tracked in Git
ADD models ./models

# for configuration parameters  
COPY params.yaml .      

# Optional: Copy DVC hidden directories, .dvc, and .dvc/config if applicable
# Copy .dvc and .git directories for DVC and Git functionality
COPY .dvc/ ./.dvc
COPY .git/ ./.git
#copy api code
COPY app.py ./app.py

# Set up environment variables
ENV MODEL_DIR=/app/models
ENV DATA_DIR=/app/data

# Pull the data from the DVC remote storage
RUN dvc pull -f

# Expose API port
EXPOSE 8005

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]

# Set the default command to execute the DVC pipeline
#CMD ["dvc", "repro"]