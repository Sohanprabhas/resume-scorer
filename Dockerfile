   #  Use a lightweight Python image
FROM python:3.10-slim

#  Set the working directory inside the container
WORKDIR /app

# Copy all files from your project into the container
COPY . .

#  Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#  Expose the port FastAPI will use
EXPOSE 8000

#  Command to start the FastAPI server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
