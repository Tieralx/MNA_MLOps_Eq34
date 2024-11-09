# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files to container
COPY models\lr_model.pkl /app/
COPY src\data_load.py /app/      
COPY src\preprocess.py /app/ 
COPY src\evaluate.py /app/   
COPY main.py /app/

# Install dependencies
RUN pip install fastapi uvicorn scikit-learn pydantic dvc

# Instala las dependencias del proyecto
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "dvc", "repro"]