# Use a lightweight official Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and the source code into the container
COPY requirements.txt .
COPY main.py .
COPY xgboost_ecommerce_pipeline.pkl .

# Install the required Python dependencies
# --no-cache-dir helps keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Inform Docker that the container listens on port 8000 at runtime
EXPOSE 8000

# Specify the command to run the application when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]