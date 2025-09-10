import mlflow
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
HF_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
LOCAL_MODEL_DIR = "distilbert-local"
ARTIFACT_PATH = "distilbert-sentiment-model"


def main():
    print(f"Downloading model '{HF_MODEL_NAME}' from Hugging Face Hub...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    print("Download complete.")

    print(f"Saving model and tokenizer to local directory: '{LOCAL_MODEL_DIR}'...")
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    model.save_pretrained(LOCAL_MODEL_DIR)
    print("Local save complete.")

    mlflow.set_experiment("SentimentAnalysis-Transformers")
    with mlflow.start_run() as run:
        print(f"Logging directory '{LOCAL_MODEL_DIR}' as an artifact to MLflow...")
        mlflow.log_artifacts(
            local_dir=LOCAL_MODEL_DIR,
            artifact_path=ARTIFACT_PATH
        )
        print("Artifact logging complete.")
        print(f"SUCCESS! New Run ID is: {run.info.run_id}")

if __name__ == "__main__":
    main()