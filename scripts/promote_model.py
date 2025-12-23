# promote model (RECTIFIED - alias based)

import os
import mlflow
from mlflow.tracking import MlflowClient


def promote_model():
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

    client = MlflowClient()

    model_name = "my_model"

    # -------------------------------
    # 1. Get version tagged as "staging"
    # -------------------------------
    staging_versions = client.get_model_version_by_alias(
        name=model_name,
        alias="staging"
    )

    staging_version = staging_versions.version

    # -------------------------------
    # 2. Remove old champion (if exists)
    # -------------------------------
    try:
        old_champion = client.get_model_version_by_alias(
            name=model_name,
            alias="champion"
        )
        client.delete_registered_model_alias(
            name=model_name,
            alias="champion"
        )
        print(f"Removed champion alias from version {old_champion.version}")
    except Exception:
        # No existing champion → safe to ignore
        pass

    # -------------------------------
    # 3. Promote staging → champion
    # -------------------------------
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=staging_version
    )

    print(f"Model version {staging_version} promoted to CHAMPION")


if __name__ == "__main__":
    promote_model()
