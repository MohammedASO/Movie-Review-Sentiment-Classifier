# 🎬 Movie Review Sentiment Classifier  

A simple machine learning project that predicts whether a movie review is **positive** or **negative**.  

--------

## 📂 Project Structure
notebooks/
sentiment_analysis.ipynb # Notebook for exploration and experiments
src/
train.py # Train and save the model pipeline
predict.py # Load pipeline and classify new reviews
requirements.txt # Dependencies
README.md # Project overview


--------

## 🛠️ Tech Stack
- **Python 3.12**
- **NLTK** → dataset (`movie_reviews`)
- **pandas** → data handling
- **scikit-learn** → TF-IDF, Logistic Regression, Naive Bayes, LinearSVC
- **joblib** → save & reload models
- **Jupyter Notebook** → experimentation

--------

## 🚀 How It Works
1. **Load data**: NLTK’s movie reviews (`pos` / `neg`).  
2. **Vectorize text**: Convert words → numbers using TF-IDF with n-grams (captures phrases like *"not good"*).  
3. **Train models**: Logistic Regression, Linear SVM, and Naive Bayes.  
4. **Evaluate**: Accuracy, Precision, Recall, F1-score.  
5. **Save best pipeline**: Model + vectorizer stored as one `.pkl` file.  
6. **Predict**: Use `predict.py` to classify new reviews interactively.  

--------

## ⚡ Quick Start
```bash
# Clone repo
git clone https://github.com/MohammedASO/Movie-Review-Sentiment-Classifier.git
cd Movie-Review-Sentiment-Classifier

# Create venv
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Train models
python src/train.py

# Predict sentiment
python src/predict.py

✨ Examples:

Type a movie review (or 'quit' to exit): The film was amazing!
Prediction: pos

Type a movie review (or 'quit' to exit): The movie was boring and too long.
Prediction: neg
