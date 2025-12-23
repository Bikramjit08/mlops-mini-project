import unittest
import os
import mlflow
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # -------------------------------
        # MLflow + DagsHub authentication
        # -------------------------------
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "Bikramjit08"
        repo_name = "mlops-mini-project"

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', data=dict(text="I love this!"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Happy' in response.data or b'Sad' in response.data,
            "Response should contain either 'Happy' or 'Sad'"
        )

if __name__ == '__main__':
    unittest.main()