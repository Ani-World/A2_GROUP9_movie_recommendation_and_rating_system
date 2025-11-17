import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split # <-- Import train_test_split
from sklearn.ensemble import RandomForestRegressor # <-- Import RandomForest

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

# --- FIX for FutureWarning ---
# Fill NaNs for safety using direct assignment (avoids inplace warning)
df['Director'] = df['Director'].fillna('Unknown')
df['Actor 1'] = df['Actor 1'].fillna('Unknown')
df['Actor 2'] = df['Actor 2'].fillna('Unknown')
df['Actor 3'] = df['Actor 3'].fillna('Unknown')
df['Year'] = df['Year'].fillna(df['Year'].median())
df['Votes'] = df['Votes'].fillna(0)
df['Duration'] = df['Duration'].fillna(df['Duration'].mean())
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
# --- END FIX ---

print(f"✅ Loaded and cleaned dataset: {len(df)} rows")

# -----------------------------
# 1. K-MEANS CLUSTERING (No Change)
# (This is good for content-based similarity)
# -----------------------------
print("\n--- Training K-Means Model ---")
# Genre features
genre_cols = [c for c in df.columns if c.startswith("Genre_")]
if not genre_cols:
    raise ValueError("No Genre_* columns found in dataset.")

# Actor 1 encoding
top_actors = df["Actor 1"].value_counts().head(50).index.tolist()
df["Actor_1_encoded"] = df["Actor 1"].apply(lambda x: x if x in top_actors else "Other")

actor_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
actor_features = actor_encoder.fit_transform(df[["Actor_1_encoded"]])
actor_feature_names = [f"Actor_{a}" for a in actor_encoder.categories_[0]]

X_genre_actor = np.hstack([df[genre_cols].fillna(0).values, actor_features])
feature_cols_cluster = genre_cols + actor_feature_names
print(f"📊 Using {len(feature_cols_cluster)} features for clustering (Genres + Actor 1)")

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_genre_actor)
n_clusters = 10 
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
kmeans.fit(X_cluster_scaled)
df["Cluster"] = kmeans.labels_
print(f"✅ KMeans trained with {n_clusters} clusters")

# -----------------------------
# 2. REGRESSION MODEL (New & Improved)
# (Using target encoding to predict ratings)
# -----------------------------
print("\n--- Training Regression Model ---")

# --- A. Split Data FIRST to Prevent Leakage ---
X = df.drop('Rating', axis=1) # All columns except Rating
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(f"Train/Test split: {len(X_train)} train, {len(X_test)} test")

# --- FIX for KeyError: Add ratings to X_train/X_test *before* using them ---
X_train['Rating_y_train'] = y_train
X_test['Rating_y_train'] = y_test
# --- END FIX ---

# --- B. Calculate Target Encoding Means (from Training Data ONLY) ---
global_mean_rating = y_train.mean()

# Calculate means, fill missing values with the global mean
dir_mean_rat = X_train.groupby('Director')['Rating_y_train'].mean().fillna(global_mean_rating)
a1_mean_rat = X_train.groupby('Actor 1')['Rating_y_train'].mean().fillna(global_mean_rating)
a2_mean_rat = X_train.groupby('Actor 2')['Rating_y_train'].mean().fillna(global_mean_rating)
a3_mean_rat = X_train.groupby('Actor 3')['Rating_y_train'].mean().fillna(global_mean_rating)

# For Genre, we can average the means of all genres present
genre_means_list = []
for col in genre_cols:
    genre_mean = X_train[X_train[col] == 1]['Rating_y_train'].mean()
    genre_means_list.append(genre_mean)
g_mean_rat_global = np.nanmean(genre_means_list) # Global mean for genres

# --- C. Feature Engineering Function (re-usable for train/test) ---
def create_regression_features(df): # No y_series needed here
    X_new = pd.DataFrame()
    X_new['Year'] = df['Year']
    X_new['Votes'] = df['Votes']
    X_new['Duration'] = df['Duration']
    
    # Apply learned means
    X_new['Dir_enc'] = df['Director'].map(dir_mean_rat).fillna(global_mean_rating)
    X_new['A1_enc'] = df['Actor 1'].map(a1_mean_rat).fillna(global_mean_rating)
    X_new['A2_enc'] = df['Actor 2'].map(a2_mean_rat).fillna(global_mean_rating)
    X_new['A3_enc'] = df['Actor 3'].map(a3_mean_rat).fillna(global_mean_rating)
    
    # Calculate weighted genre mean for each movie
    g_means = df[genre_cols].copy()
    for i, col in enumerate(genre_cols):
        g_means[col] = g_means[col] * genre_means_list[i]
    # Replace 0s with NaN to avoid skewing the mean, then calc mean across row
    X_new['G_mean_rat'] = g_means.replace(0, np.nan).mean(axis=1).fillna(g_mean_rat_global)
    
    return X_new

# Create regression features for train and test sets
X_reg_train = create_regression_features(X_train)
X_reg_test = create_regression_features(X_test)

# --- D. Train the Regressor ---
# We use RandomForest, it's powerful and less prone to overfitting than a single Decision Tree
regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=5)
regressor.fit(X_reg_train, y_train)

# --- E. Evaluate (This is your *real* R2 score) ---
r2_score = regressor.score(X_reg_test, y_test)
print(f"✅ New Regressor (RandomForest) trained.")
print(f"📊 Model R2 Score (Test Set): {r2_score:.4f}")

# -----------------------------
# Save Models & Mappings
# -----------------------------

# --- A. Save K-Means (No Change) ---
joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
joblib.dump(scaler_cluster, os.path.join(MODEL_DIR, "scaler_cluster.joblib"))

# --- B. Save New Regressor ---
joblib.dump(regressor, os.path.join(MODEL_DIR, "regressor_model.pkl"))

# --- C. Save the Target Encoding "Scaler" (which is just a dict of means) ---
# The backend will load this file to apply encoding to new/unseen movies
target_encoder_means = {
    'global_mean_rating': global_mean_rating,
    'g_mean_rat_global': g_mean_rat_global,
    'genre_means_list': genre_means_list,
    'dir_mean_rat': dir_mean_rat.to_dict(), # Convert pandas Series to dict
    'a1_mean_rat': a1_mean_rat.to_dict(),
    'a2_mean_rat': a2_mean_rat.to_dict(),
    'a3_mean_rat': a3_mean_rat.to_dict()
}
# We save this as 'scaler.joblib' to match the file your backend is already loading
# (Even though it's a dict, joblib can handle it)
joblib.dump(target_encoder_means, os.path.join(MODEL_DIR, "scaler.joblib"))

# --- D. Save Metadata (feature_cols.json) ---
# Update with the *new* regression feature names
feature_cols_regression_new = X_reg_train.columns.tolist()
metadata = {
    "cluster_feature_cols": feature_cols_cluster,
    "regression_feature_cols": feature_cols_regression_new, # <-- Updated
    "actor_encoder_categories": actor_encoder.categories_[0].tolist(),
    "genre_cols_list": genre_cols # <-- Add list of genres
}
with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# --- E. Save Name-to-Feature Mapping (No Change) ---
# This file is still used by your backend to build the features array
# We need to add the new numeric/encoded cols to it
df_with_new_features = create_regression_features(df)
df = pd.concat([df, df_with_new_features], axis=1)

# Ensure all columns exist before saving
cols_to_save = ["Name", "Actor 1", "Cluster"] + genre_cols + feature_cols_regression_new
# Filter to only columns that actually exist in the final dataframe
cols_to_save = [col for col in cols_to_save if col in df.columns]

df_out = df[cols_to_save].copy()
df_out.to_csv(os.path.join(MODEL_DIR, "name_features_mapping.csv"), index=False)
print("✅ All models and feature mappings saved to Backend/models")

# -----------------------------
# Cluster overview (quick check)
# -----------------------------
print("\n📊 Cluster distribution:")
print(df["Cluster"].value_counts().sort_index())
print("\n🚀 Training complete!")