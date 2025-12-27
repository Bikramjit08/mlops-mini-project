from flask import Flask, render_template, request
import mlflow
import dagshub
import pickle
import os
import pandas as pd
import nltk
import string
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Check if we're in test mode
TESTING_MODE = os.getenv('TESTING_MODE', 'false').lower() == 'true'




def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text


if not TESTING_MODE:
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")
    
    print(f"✓ Token found (length: {len(dagshub_token)})")
    
    # Authenticate with DagsHub using token - THIS IS THE KEY LINE
    try:
        print("Authenticating with DagsHub token...")
        dagshub.auth.add_app_token(token=dagshub_token)
        print("✓ DagsHub token authentication successful")
    except Exception as e:
        print(f"✗ DagsHub token authentication failed: {e}")
        raise
    
    # NOW initialize DagsHub (after authentication)
    try:
        print("Initializing DagsHub...")
        dagshub.init(repo_owner='Bikramjit08', repo_name='mlops-mini-project', mlflow=True)
        print("✓ DagsHub initialized successfully")
    except Exception as e:
        print(f"✗ DagsHub initialization failed: {e}")
        raise
    
    # Set MLflow credentials
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    
    # Set MLflow tracking URI
    tracking_uri = 'https://dagshub.com/Bikramjit08/mlops-mini-project.mlflow'
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✓ MLflow tracking URI set: {tracking_uri}")
    
    # Test MLflow connection
    try:
        print("Testing MLflow connection...")
        client = mlflow.MlflowClient()
        experiments = client.search_experiments(max_results=1)
        print(f"✓ MLflow connection successful!")
    except Exception as e:
        print(f"✗ MLflow connection failed: {e}")
        raise
    
    # Load model from model registry
    try:
        print("Loading model from registry...")
        def get_latest_model_version(model_name):
            client = mlflow.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["Production"])
            if not latest_version:
                latest_version = client.get_latest_versions(model_name, stages=["None"])
            return latest_version[0].version if latest_version else None

        model_name = "my_model"
        model_version = get_latest_model_version(model_name)
        
        if not model_version:
            raise ValueError(f"No model version found for {model_name}")
        
        print(f"✓ Found model version: {model_version}")
        model_uri = f'models:/{model_name}/{model_version}'
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        raise
    
    # Load vectorizer
    try:
        print("Loading vectorizer...")
        vectorizer_path = 'models/vectorizer.pkl'
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
        
        vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        print("✓ Vectorizer loaded successfully")
    except Exception as e:
        print(f"✗ Vectorizer loading failed: {e}")
        raise
    
    print("\n🎉 All components loaded successfully!\n")
else:
    # Mock model and vectorizer for testing
    model = None
    vectorizer = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if TESTING_MODE:
        # Return mock prediction for tests
        return render_template('index.html', result='Happy')
    
    text = request.form['text']
    text = normalize_text(text)
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
    result = model.predict(features_df)
    return render_template('index.html', result=result[0])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")