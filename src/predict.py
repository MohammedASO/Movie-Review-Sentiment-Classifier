# src/predict.py
import joblib
from pathlib import Path

def load_pipeline():
    path = Path(__file__).parent / "sentiment_pipeline.pkl"
    return joblib.load(path)

def predict_review(text: str):
    pipe = load_pipeline()
    return pipe.predict([text])[0]

if __name__ == "__main__":
    while True:
        sample = input("Type a movie review (or 'quit' to exit): ")
        if sample.lower().strip() == "quit":
            break
        print("Prediction:", predict_review(sample))
