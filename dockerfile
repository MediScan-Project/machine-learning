FROM python:3.9

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Set the working directory inside the container
WORKDIR /machine-learning

# Copy the requirements file into the container
COPY requirements.txt .

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

RUN apt update && apt install -y libgl1-mesa-glx


COPY . .

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "server_minimal:app", "--host", "0.0.0.0", "--port", "8000"]