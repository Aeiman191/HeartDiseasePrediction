# Use the official Python image from the Docker Hub
FROM python:3.10.5

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project files into the container
COPY . .

# Expose the port on which your Flask app will run
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "app.py"]
