# src/train.py
import argparse
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# ---------- Data loading ----------
def load_movie_reviews_df() -> pd.DataFrame:
    nltk.download("movie_reviews", quiet=True)
    from nltk.corpus import movie_reviews

    reviews, labels = [], []
    for cat in movie_reviews.categories():  # 'pos' / 'neg'
        for fid in movie_reviews.fileids(cat):
            reviews.append(movie_reviews.raw(fid))
            labels.append(cat)
    return pd.DataFrame({"review": reviews, "label": labels})

# ---------- Model candidates ----------
def build_candidates(
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2,
    max_df=0.9,
):
    common_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        lowercase=True,
        strip_accents="unicode",
    )

    candidates = {
        "logreg": Pipeline([
            ("tfidf", common_vectorizer),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None)),
        ]),
        "linearsvc": Pipeline([
            ("tfidf", common_vectorizer),
            ("clf", LinearSVC()),
        ]),
        "nb": Pipeline([
            ("tfidf", common_vectorizer),
            ("clf", MultinomialNB()),
        ]),
    }
    return candidates

# ---------- Training & evaluation ----------
def evaluate_models(X_train, y_train, candidates, cv_splits=5, scoring="f1_macro"):
    """Cross-validate each candidate and return a sorted list of (name, mean, std)."""
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    results = []
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring=scoring, n_jobs=None)
        results.append((name, scores.mean(), scores.std()))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def train_best_and_report(X_train, X_test, y_train, y_test, candidates, best_name):
    best_pipeline = candidates[best_name]
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    print("=" * 60)
    print(f"Best model: {best_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=["neg", "pos"])
    print("Confusion matrix (rows=true, cols=pred) with labels [neg, pos]:")
    print(cm)
    print("=" * 60)
    return best_pipeline

# ---------- Optional: show top features for LR/LinearSVC ----------
def show_top_features(pipeline, k=10):
    try:
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        vocab = np.array(tfidf.get_feature_names_out())

        if hasattr(clf, "coef_"):
            coefs = clf.coef_[0]
            top_pos = np.argsort(coefs)[-k:][::-1]
            top_neg = np.argsort(coefs)[:k]
            print("\nTop positive features:")
            print(vocab[top_pos])
            print("\nTop negative features:")
            print(vocab[top_neg])
        else:
            # NB/others may not have direct signed coefficients in the same way
            pass
    except Exception:
        pass

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Train sentiment classifier on NLTK movie_reviews.")
    p.add_argument("--test_size", type=float, default=0.2, help="Test fraction (default 0.2)")
    p.add_argument("--ngram_max", type=int, default=2, help="Max n-gram size (default 2)")
    p.add_argument("--max_features", type=int, default=20000, help="TF-IDF max features (default 20000)")
    p.add_argument("--min_df", type=int, default=2, help="TF-IDF min_df (default 2)")
    p.add_argument("--max_df", type=float, default=0.9, help="TF-IDF max_df (default 0.9)")
    p.add_argument("--models", type=str, default="logreg,linearsvc,nb",
                   help="Comma-separated subset to try: logreg,linearsvc,nb")
    p.add_argument("--scoring", type=str, default="f1_macro", help="CV scoring metric (default f1_macro)")
    p.add_argument("--cv", type=int, default=5, help="CV splits (default 5)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    p.add_argument("--save_path", type=str, default="src/sentiment_pipeline.pkl",
                   help="Where to save the best pipeline (default src/sentiment_pipeline.pkl)")
    p.add_argument("--show_topk", type=int, default=0, help="Show top-k features for linear models (0 disables)")
    return p.parse_args()

def main():
    args = parse_args()

    # Load data
    df = load_movie_reviews_df()
    X = df["review"].values
    y = df["label"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Build candidates
    candidates = build_candidates(
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
    )

    # Filter which to try
    wanted = {k.strip().lower() for k in args.models.split(",")}
    candidates = {name: pipe for name, pipe in candidates.items() if name in wanted}
    if not candidates:
        raise ValueError("No valid models selected. Choose from: logreg,linearsvc,nb")

    # CV evaluate
    print("Cross-validating candidates...")
    results = evaluate_models(X_train, y_train, candidates, cv_splits=args.cv, scoring=args.scoring)
    for name, mean, std in results:
        print(f"  {name:9s}  {args.scoring}: {mean:.4f} ± {std:.4f}")

    best_name = results[0][0]
    # Fit best and report on held-out test set
    best_pipeline = train_best_and_report(X_train, X_test, y_train, y_test, candidates, best_name)

    # Optionally show top features (only meaningful for linear models with coef_)
    if args.show_topk > 0:
        show_top_features(best_pipeline, k=args.show_topk)

    # Save best pipeline
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, save_path)
    print(f"\nSaved best pipeline '{best_name}' to: {save_path.resolve()}")

if __name__ == "__main__":
    main()
