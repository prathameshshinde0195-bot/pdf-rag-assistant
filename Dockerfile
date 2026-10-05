# Base Python 3.11 image
FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=0.0.0.0:11434

# Install system dependencies & curl to install Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama binary
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create startup script to start Ollama daemon, pull models, and launch Streamlit
RUN echo '#!/bin/bash\n\
ollama serve &\n\
sleep 5\n\
echo "Pulling fast models in cloud container..."\n\
ollama pull llama3.2:1b || ollama pull llama3.2:3b\n\
ollama pull nomic-embed-text\n\
echo "Starting Streamlit..."\n\
streamlit run app.py --server.port=7860 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose Streamlit default port for Hugging Face Spaces
EXPOSE 7860

# Start container
CMD ["/app/start.sh"]
