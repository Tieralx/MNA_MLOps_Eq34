# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files to container
COPY cervical_cancer_model.pkl /app/
COPY main.py /app/
COPY sobar-72.csv /app/

# Install dependencies
RUN pip install fastapi uvicorn scikit-learn pydantic pandas

# Expose port
EXPOSE 8003 

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
