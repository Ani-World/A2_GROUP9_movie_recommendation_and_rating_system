import sqlite3
import pandas as pd
import json
import os
from datetime import datetime

# === 1. Paths ===
csv_path = "datasets/movies_data.csv"
db_dir = "instance"
db_path = os.path.join(db_dir, "movies.db")
os.makedirs(db_dir, exist_ok=True)

# === 2. Read CSV ===
df = pd.read_csv(csv_path)

# Normalize column names
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# === 3. Create genre_vector JSON from Genre column ===
def make_genre_vector(genre_str):
    if pd.isna(genre_str):
        return json.dumps({})
    genres = [g.strip() for g in genre_str.split(",")]
    return json.dumps({g: 1 for g in genres})

df["genre_vector"] = df["Genre"].apply(make_genre_vector)

# === 4. Add placeholder columns ===
df["cluster_id"] = None
df["feature_path"] = None
df["poster_url"] = None

# === 5. Select + rename columns to match target schema ===
movies_df = df.rename(
    columns={
        "Name": "name",
        "Year": "year",
        "Duration": "duration",
        "Rating": "avg_rating",
        "Votes": "votes",
        "Director": "director",
        "Actor_1": "actor1",
        "Actor_2": "actor2",
        "Actor_3": "actor3"
    }
)[[
    "name",
    "year",
    "duration",
    "avg_rating",
    "votes",
    "director",
    "actor1",
    "actor2",
    "actor3",
    "genre_vector",
    "cluster_id",
    "feature_path",
    "poster_url"
]]

# === 6. Connect to SQLite ===
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# === 7. Create tables ===

# Movies table
cursor.execute("""
CREATE TABLE IF NOT EXISTS movies (
    movie_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    year INTEGER,
    duration INTEGER,
    avg_rating REAL,
    votes INTEGER,
    director TEXT,
    actor1 TEXT,
    actor2 TEXT,
    actor3 TEXT,
    genre_vector JSON,
    cluster_id INTEGER,
    feature_path TEXT,
    poster_url TEXT
);
""")

# Users table
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS users (
#     user_id INTEGER PRIMARY KEY AUTOINCREMENT,
#     name TEXT,
#     email TEXT UNIQUE,
#     created_at DATETIME DEFAULT CURRENT_TIMESTAMP
# );
# """)

# Ratings table
cursor.execute("""
CREATE TABLE IF NOT EXISTS ratings (
    user_id INTEGER,
    movie_id INTEGER,
    rating REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, movie_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);
""")

conn.commit()

# === 8. Insert movies ===
movies_df.to_sql("movies", conn, if_exists="replace", index=False)

conn.commit()
conn.close()

print(f"✅ Database created at {db_path} with:")
print(f"   - {len(movies_df)} movies")
#print("   - users table ready")
print("   - ratings table ready")
