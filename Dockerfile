FROM python:3.10-slim

WORKDIR /app

# Copy requirements FIRST (for better Docker layer caching)
COPY flask_app/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Verify scikit-learn version (for debugging)
RUN python -c "import sklearn; print(f'✓ scikit-learn: {sklearn.__version__}')"

# Copy Flask app code
COPY flask_app/ /app/

# Copy model file
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
