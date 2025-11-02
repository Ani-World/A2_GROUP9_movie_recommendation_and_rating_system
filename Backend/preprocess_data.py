import pandas as pd
import numpy as np
import os

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv("datasets/movies_data.csv")

# -----------------------------
# Handle genres
# -----------------------------
genre_set = set()
for g in df['Genre'].dropna():
    genre_set.update([x.strip() for x in g.split(',')])

genre_list = sorted(genre_set)
for genre in genre_list:
    df[f"Genre_{genre}"] = df['Genre'].apply(lambda x: int(genre in x) if pd.notna(x) else 0)

df['genre_list'] = df['Genre'].apply(lambda x: [g.strip() for g in x.split(',')] if pd.notna(x) else [])

# -----------------------------
# Normalize numeric features
# -----------------------------
df['Year_norm'] = (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min())
df['Duration_norm'] = df['Duration'] / df['Duration'].max()
df['Rating_norm'] = df['Rating'] / 10.0
df['Votes_log'] = np.log1p(df['Votes'])

# -----------------------------
# Combine numeric + genre for ML
# -----------------------------
feature_cols = [f"Genre_{g}" for g in genre_list] + ['Year_norm', 'Duration_norm', 'Rating_norm', 'Votes_log']
X = df[feature_cols].values

# -----------------------------
# Prepare Apriori-friendly data
# -----------------------------
df['actors_list'] = df[['Actor 1', 'Actor 2', 'Actor 3']].apply(
    lambda x: [a.strip() for a in x if pd.notna(a)], axis=1
)
df['director_list'] = df['Director'].apply(lambda x: [x.strip()] if pd.notna(x) else [])

df['apriori_items'] = df.apply(lambda row: row['genre_list'] + row['actors_list'] + row['director_list'], axis=1)
apriori_transactions = df['apriori_items'].tolist()

# -----------------------------
# Save processed CSV
# -----------------------------
os.makedirs("datasets/processed", exist_ok=True)
processed_csv_path = "datasets/processed/movies_data_processed.csv"
df.to_csv(processed_csv_path, index=False)
print(f"Processed CSV saved to {processed_csv_path}")

# -----------------------------
# Optional inspection
# -----------------------------
print("Feature matrix shape:", X.shape)
print("Example feature row:", X[0])
print("Apriori transaction example:", apriori_transactions[0])
