FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY streamlit_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY streamlit_app/ .

# Create data directory structure
RUN mkdir -p data

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
