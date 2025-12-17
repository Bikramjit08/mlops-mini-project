from flask import Flask,render_template,request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import pickle


dagshub_url = "https://dagshub.com"
repo_owner = "Bikramjit08"
repo_name = "mlops-mini-project"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
dagshub.init(repo_owner='Bikramjit08', repo_name='mlops-mini-project', mlflow=True)

app = Flask(__name__)

# load model from model registry
model_name = "my_model"
model_version = 1

model_uri = f'models:/{model_name}/{model_version}'

model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))


@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict ():
    text = request.form['text']




    # clean the text

    text = normalize_text(text)

    # Apply BOW
    features = vectorizer.transform([text])

    # Prediction
    result = model.predict(features)



    # Show
    return render_template('index.html', result=result[0])



    


app.run(debug=True)