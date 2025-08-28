# Use a slim Python image to keep the container size small
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install gcloud CLI
RUN apt-get update && apt-get install -y curl gnupg dnsutils \
  && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
  && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
  && apt-get update && apt-get install -y google-cloud-sdk \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/main.py .

# Copy entrypoint script
COPY src/entrypoint-clean.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose the port the app runs on
EXPOSE 8000

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
