import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

# Clean text to remove noise and punctuation
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def load_data():
    print("📥 Loading text data...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train["catalog_content"] = train["catalog_content"].astype(str).apply(clean_text)
    test["catalog_content"] = test["catalog_content"].astype(str).apply(clean_text)

    return train, test, train["catalog_content"].tolist(), test["catalog_content"].tolist()

def build_tfidf(train_texts, test_texts, max_features=50000, n_svd=200):
    print("⚙️ Building TF-IDF vectors...")
    all_texts = train_texts + test_texts
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    tfidf = vectorizer.fit_transform(all_texts)

    print("⚙️ Applying SVD dimensionality reduction...")
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    reduced = svd.fit_transform(tfidf)

    train_features = reduced[:len(train_texts)]
    test_features = reduced[len(train_texts):]

    joblib.dump(train_features, "features/train_text_features.pkl")
    joblib.dump(test_features, "features/test_text_features.pkl")
    joblib.dump(vectorizer, "features/tfidf_vectorizer.pkl")
    joblib.dump(svd, "features/svd_model.pkl")

    print(f"✅ Text features extracted: Train {train_features.shape}, Test {test_features.shape}")
    print("💾 Features saved in the 'features' folder.")

if __name__ == "__main__":
    train, test, train_texts, test_texts = load_data()
    build_tfidf(train_texts, test_texts)
