# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.linear_model import LinearRegression
# import joblib
# import os

# # Load clean dataset
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(BASE_DIR, "..", "datasets", "movies_data_processed.csv")
# df = pd.read_csv(data_path)

# # Example feature matrix
# X = df[["Year", "Duration", "Rating", "Votes"]].fillna(0)

# # Train KMeans
# kmeans = KMeans(n_clusters=4, random_state=42)
# kmeans.fit(X)
# joblib.dump(kmeans, "Backend/models/kmeans_model.pkl")

# # Train Linear Regressor
# regressor = LinearRegression()
# regressor.fit(X, df["Rating"])
# joblib.dump(regressor, "Backend/models/regressor_model.pkl")

# print("✅ Models saved: kmeans_model.pkl, regressor_model.pkl")
import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "datasets", "movies_data_processed.csv")

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# -----------------------------
# Feature matrix for clustering
# -----------------------------
cluster_features = ["Year", "Duration", "Rating", "Votes"]
X = df[cluster_features].fillna(0)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Train Linear Regressor
regressor = LinearRegression()
regressor.fit(X_scaled, df["Rating"].fillna(0))

# Save models
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(kmeans, os.path.join(model_dir, "kmeans_model.pkl"))
joblib.dump(regressor, os.path.join(model_dir, "regressor_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

print("✅ Models saved: kmeans_model.pkl, regressor_model.pkl, scaler.joblib")

# -----------------------------
# Prepare files for cluster-based recommendation
# -----------------------------
feature_cols = cluster_features
with open(os.path.join(model_dir, "feature_cols.json"), "w") as f:
    json.dump(feature_cols, f)

# Save mapping of movie names to features
name_feat = df[["Name"] + cluster_features].copy()
name_feat.to_csv(os.path.join(model_dir, "name_features_mapping.csv"), index=False)
print("✅ Exported cluster recommender mapping: name_features_mapping.csv")

# -----------------------------
# Prepare for content-based recommender
# -----------------------------
def build_genre_matrix(df):
    genre_cols = [c for c in df.columns if c.startswith("Genre_")]
    if genre_cols:
        print(f"Using {len(genre_cols)} Genre_* columns for content features.")
        return df[genre_cols].fillna(0).astype(int).values, "genres", genre_cols
    elif "Genre" in df.columns:
        print("Building Genre_* columns from text in 'Genre'.")
        genres = df["Genre"].fillna("").astype(str).str.get_dummies(sep=",")
        return genres.values, "genres", list(genres.columns)
    elif any(c in df.columns for c in ["Plot", "Description", "Overview"]):
        text_col = next(c for c in ["Plot", "Description", "Overview"] if c in df.columns)
        print(f"Using TF-IDF on '{text_col}' for content features.")
        tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
        feature_matrix = tfidf.fit_transform(df[text_col].fillna("").astype(str)).toarray()
        return feature_matrix, "text", [f"tfidf_{i}" for i in range(feature_matrix.shape[1])]
    else:
        raise ValueError("No usable Genre_* or text columns found for content-based recommender.")

try:
    feature_matrix, feature_type, used_cols = build_genre_matrix(df)
    np.save(os.path.join(model_dir, "content_features.npy"), feature_matrix)
    with open(os.path.join(model_dir, "content_features_meta.json"), "w") as f:
        json.dump({"feature_type": feature_type, "columns": used_cols}, f, indent=2)
    print(f"✅ Content-based features saved ({feature_type}, {feature_matrix.shape[1]} columns)")
except Exception as e:
    print("⚠️ Content-based recommender not built:", e)

print("\n🎬 All models and feature mappings prepared successfully.")
