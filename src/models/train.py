# src/models/train.py

import argparse
import joblib
import os
import pandas as pd
import mlflow
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000' 
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'

def main(args: argparse.Namespace) -> None:
    """
    Основная функция для загрузки данных, обучения модели
    и логирования эксперимента в MLflow.
    """
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    
    mlflow.set_experiment("SentimentAnalysis")

    with mlflow.start_run():
        print("Starting training run...")
        mlflow.log_param("model_type", "SGDClassifier")
        print("Loading data...")
        train_df = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
        val_df = pd.read_csv(os.path.join(args.data_path, 'validation.csv'))

        train_df.dropna(subset=['cleaned_text'], inplace=True)
        val_df.dropna(subset=['cleaned_text'], inplace=True)

        X_train, y_train = train_df['cleaned_text'], train_df['sentiment']
        X_val, y_val = val_df['cleaned_text'], val_df['sentiment']


        print("\n" + "="*50)
        print("DATA DIAGNOSTICS BEFORE TRAINING")
        print("="*50)
        
        print("\n[1] Class distribution in TRAINING data (y_train):")
        print(y_train.value_counts())
        
        print("\n[2] Class distribution in VALIDATION data (y_val):")
        print(y_val.value_counts())
        
        print("\n[3] Sample of TRAINING data:")
        print(train_df.head(3))
        
        print("\n[4] Sample of VALIDATION data:")
        print(val_df.head(3))

        print("\n" + "="*50)
        print("END OF DIAGNOSTICS")
        print("="*50 + "\n")
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('sgd', SGDClassifier(loss='log_loss', random_state=42, class_weight='balanced'))
        ])
        
        print("Training model...")
        pipeline.fit(X_train, y_train)



        print("\n" + "="*50)
        print("MODEL INTERROGATION AFTER TRAINING")
        print("="*50)

        test_phrases = [
            "this is the best laptop i have ever owned amazing product", # Ожидаем 1
            "i love this it works perfectly and looks great",         # Ожидаем 1
            "terrible quality it broke after just one week",           # Ожидаем 0
            "a piece of junk stopped working almost immediately"       # Ожидаем 0
        ]
        
        test_data = pd.DataFrame({"cleaned_text": test_phrases})
        
        predictions = pipeline.predict(test_data)
        
        print("\nPredictions on test phrases:")
        for phrase, pred in zip(test_phrases, predictions):
            print(f"  - Prediction: {pred} -> Phrase: '{phrase[:50]}...'")

        print("\n" + "="*50)
        print("END OF INTERROGATION")
        print("="*50 + "\n")
    

        print("Evaluating model...")
        y_pred = pipeline.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)



        print("Saving model locally...")
        model_output_dir = "sentiment-model"
        os.makedirs(model_output_dir, exist_ok=True)
        model_path = os.path.join(model_output_dir, "model.joblib")
        joblib.dump(pipeline, model_path)
        print(f"Model saved to {model_path}")

        print("Logging model artifact to MLflow...")
        mlflow.log_artifacts(model_output_dir, artifact_path="model")




        print("Training run finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model.")
    parser.add_argument(
        '--data-path', 
        type=str, 
        required=True, 
        help='Path to the processed data directory (output of preprocess stage).'
    )
    
    args = parser.parse_args()
    main(args)