# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# NOTE: data/best.pt (image model), data/feature_schema.json, and
# data/lifestyle_model.pkl are already included by "COPY . ." above.
# Train the lifestyle model BEFORE building this image, or the Survey/
# Combined tabs will report the model as unavailable (Image Only still
# works):
#   python training/generate_synthetic_lifestyle_data.py
#   python training/train_lifestyle_model.py

# Make port 7860 available to the world outside this container
EXPOSE 7860


# Define environment variable
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run the application
CMD ["python", "app/gradio-app.py"]

# Your Gradio app should now be accessible at http://localhost:7860
