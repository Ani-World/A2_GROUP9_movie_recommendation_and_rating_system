"""
train_kmeans_and_regressor.py
----------------------------------
Builds KMeans clusters using:
 - Genre_* one-hot columns (semantic)
 - Actor 1 encoded as one-hot (cast-based)
Also fits a regression model (using numeric + genre).

Outputs:
 - Backend/models/kmeans_model.pkl
 - Backend/models/regressor_model.pkl
 - Backend/models/scaler.joblib
 - Backend/models/feature_cols.json
 - Backend/models/name_features_mapping.csv
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "datasets", "movies_data_processed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

# -----------------------------
# Feature selection for clustering
# -----------------------------
# Genre features
genre_cols = [c for c in df.columns if c.startswith("Genre_")]
if not genre_cols:
    raise ValueError("No Genre_* columns found in dataset.")

# Actor 1 encoding
if "Actor 1" not in df.columns:
    raise ValueError("Missing 'Actor 1' column in dataset.")

# One-hot encode top actors to avoid too many columns
top_actors = df["Actor 1"].value_counts().head(50).index.tolist()
df["Actor_1_encoded"] = df["Actor 1"].apply(lambda x: x if x in top_actors else "Other")

actor_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
actor_features = actor_encoder.fit_transform(df[["Actor_1_encoded"]])
actor_feature_names = [f"Actor_{a}" for a in actor_encoder.categories_[0]]

# Combine Genre + Actor features
X_genre_actor = np.hstack([df[genre_cols].fillna(0).values, actor_features])
feature_cols_cluster = genre_cols + actor_feature_names

print(f"📊 Using {len(feature_cols_cluster)} features for clustering (Genres + Actor 1)")

# -----------------------------
# Scale and Cluster
# -----------------------------
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_genre_actor)

n_clusters = 10  # genre+actor groups, can tune later
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
kmeans.fit(X_cluster_scaled)

df["Cluster"] = kmeans.labels_
print(f"✅ KMeans trained with {n_clusters} clusters")

# -----------------------------
# Regression Model (for backend)
# -----------------------------
# Use both genre and numeric info
numeric_cols = ["Year_norm", "Duration_norm", "Rating_norm", "Votes_log"]
feature_cols_regression = genre_cols + numeric_cols
X_reg = df[feature_cols_regression].fillna(0)
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

regressor = LinearRegression()
regressor.fit(X_reg_scaled, df["Rating"].fillna(df["Rating"].mean()))
print("✅ Linear regression model trained for rating prediction")

# -----------------------------
# Save Models
# -----------------------------
joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
joblib.dump(regressor, os.path.join(MODEL_DIR, "regressor_model.pkl"))
joblib.dump(scaler_cluster, os.path.join(MODEL_DIR, "scaler_cluster.joblib"))
joblib.dump(scaler_reg, os.path.join(MODEL_DIR, "scaler.joblib"))  # for regression

# Save metadata
metadata = {
    "cluster_feature_cols": feature_cols_cluster,
    "regression_feature_cols": feature_cols_regression,
    "actor_encoder_categories": actor_encoder.categories_[0].tolist(),
}
with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# Save name-to-feature mapping
df_out = df[["Name", "Actor 1", "Cluster"] + genre_cols].copy()
df_out.to_csv(os.path.join(MODEL_DIR, "name_features_mapping.csv"), index=False)
print("✅ All models and feature mappings saved to Backend/models")

# -----------------------------
# Cluster overview (quick check)
# -----------------------------
print("\n📊 Cluster distribution:")
print(df["Cluster"].value_counts().sort_index())

print("\n🎬 Example movies per cluster:")
for c in range(min(10, n_clusters)):
    samples = df[df["Cluster"] == c]["Name"].head(3).tolist()
    print(f"Cluster {c}: {samples}")
