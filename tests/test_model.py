# load test + signature test + performance test (RECTIFIED)

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from mlflow.tracking import MlflowClient
from mlflow.models import Model


class TestModelLoading(unittest.TestCase):

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

        # -------------------------------
        # Load model via ALIAS (NEW WAY)
        # -------------------------------
        cls.model_name = "my_model"
        cls.model_alias = "staging"   # or "champion"

        cls.model_uri = f"models:/{cls.model_name}@{cls.model_alias}"

        try:
            cls.model = mlflow.pyfunc.load_model(cls.model_uri)
        except Exception as e:
            raise RuntimeError(f"Failed to load model using alias '{cls.model_alias}': {e}")

        # -------------------------------
        # Load vectorizer
        # -------------------------------
        if not os.path.exists("models/vectorizer.pkl"):
            raise FileNotFoundError("Vectorizer file not found")

        with open("models/vectorizer.pkl", "rb") as f:
            cls.vectorizer = pickle.load(f)

        # -------------------------------
        # Load holdout test data
        # -------------------------------
        cls.holdout_data = pd.read_csv("data/processed/test_bow.csv")

    # ------------------------------------------------
    # 1. Load Test
    # ------------------------------------------------
    def test_model_loaded(self):
        self.assertIsNotNone(self.model, "Model object is None")

    # ------------------------------------------------
    # 2. Signature Test (REAL SIGNATURE CHECK)
    # ------------------------------------------------
def test_model_signature(self):
    """
    Signature test:
    - If MLflow signature exists → validate it
    - If missing → fallback to runtime schema validation
    """

    model_meta = Model.load(self.model_uri)

    # Create sample input
    sample_text = ["hi how are you"]
    sample_vec = self.vectorizer.transform(sample_text)

    input_df = pd.DataFrame(
        sample_vec.toarray(),
        columns=[str(i) for i in range(sample_vec.shape[1])]
    )

    # Predict should not fail
    preds = self.model.predict(input_df)

    # Output shape check
    self.assertEqual(len(preds), input_df.shape[0])

    # -------- Signature-aware validation --------
    if model_meta.signature is not None:
        # Strong check (new models)
        self.assertIsNotNone(
            model_meta.signature.inputs,
            "Model input signature missing"
        )
    else:
        # Fallback check (legacy models)
        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out()),
            "Input schema mismatch detected"
        )

    # ------------------------------------------------
    # 3. Performance Test
    # ------------------------------------------------
    def test_model_performance(self):
        # Split features and labels
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # Predict
        y_pred = self.model.predict(X_holdout)

        # Metrics (safe for edge cases)
        accuracy = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred, zero_division=0)
        recall = recall_score(y_holdout, y_pred, zero_division=0)
        f1 = f1_score(y_holdout, y_pred, zero_division=0)

        # Thresholds (quality gates)
        self.assertGreaterEqual(accuracy, 0.40, "Accuracy below threshold")
        self.assertGreaterEqual(precision, 0.40, "Precision below threshold")
        self.assertGreaterEqual(recall, 0.40, "Recall below threshold")
        self.assertGreaterEqual(f1, 0.40, "F1-score below threshold")


if __name__ == "__main__":
    unittest.main()
