# Use lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY app.py .
COPY requirements.txt .
COPY tfidf_vectorizer.pkl .
COPY tfidf_matrix.pkl .
COPY product_data.pkl .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Cloud Run
EXPOSE 8080

# Start FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
