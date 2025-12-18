# register_model.py

import json
import mlflow
import logging
import os
import dagshub

# =====================================================
# DagsHub & MLflow Configuration
# =====================================================



# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Bikramjit08"
repo_name = "mlops-mini-project"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')




# =====================================================
# Logging Configuration
# =====================================================

logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_registration_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =====================================================
# Helper Functions
# =====================================================

def load_model_info(file_path: str) -> dict:
    """Load model run information from a JSON file."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logger.debug("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError:
        logger.error("Model info file not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading model info: %s", e)
        raise


def register_model(model_name: str, model_info: dict) -> None:
    """Register model in MLflow Model Registry using aliases."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug("Registering model from URI: %s", model_uri)

        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        client = mlflow.tracking.MlflowClient()

        # Use alias instead of deprecated stages
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version
        )

        logger.info(
            "Model '%s' version %s registered and assigned alias 'staging'",
            model_name,
            model_version.version
        )

    except Exception as e:
        logger.error("Error during model registration: %s", e)
        raise

# =====================================================
# Main Execution
# =====================================================

def main() -> None:
    try:
        model_info_path = "reports/experiment_info.json"
        model_name = "my_model"

        model_info = load_model_info(model_info_path)
        register_model(model_name, model_info)

        logger.info("Model registration completed successfully.")

    except Exception as e:
        logger.error("Model registration process failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
