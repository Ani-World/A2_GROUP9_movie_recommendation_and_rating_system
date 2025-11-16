# import sqlite3
# import requests
# import time
# from itertools import cycle

# # === 1. Database ===
# DB_PATH = "instance/movies.db"

# # === 2. Your OMDB API keys (replace with your actual keys) ===
# OMDB_KEYS = [
#     "",
#     "",
#     "",
#     "",
# ]

# # Rotate keys infinitely
# api_keys = cycle(OMDB_KEYS)

# # === 3. Helper function to fetch poster ===
# def fetch_poster(movie_name, api_key):
#     url = "https://www.omdbapi.com/"
#     try:
#         response = requests.get(url, params={"t": movie_name, "apikey": api_key}, timeout=5)
#         data = response.json()
#         poster = data.get("Poster")
#         if poster and poster != "N/A":
#             return poster
#     except Exception as e:
#         print(f"⚠️ Error fetching '{movie_name}': {e}")
#     # fallback placeholder
#     return f"https://via.placeholder.com/300x420/1a1a2e/ffffff?text={movie_name.replace(' ', '+')}"

# # === 4. Update movies in batches of 1000 ===
# BATCH_SIZE = 1000

# def update_posters():
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()

#     # Count total movies
#     cursor.execute("SELECT COUNT(*) FROM movies")
#     total_movies = cursor.fetchone()[0]
#     print(f"Total movies in DB: {total_movies}")

#     for offset in range(0, total_movies, BATCH_SIZE):
#         cursor.execute(
#             "SELECT movie_id, name, poster_url FROM movies ORDER BY movie_id LIMIT ? OFFSET ?",
#             (BATCH_SIZE, offset)
#         )
#         batch = cursor.fetchall()
#         print(f"Processing batch {offset} → {offset + len(batch)}")

#         for movie_id, name, poster_url in batch:
#             # Skip if already has a valid poster
#             if poster_url and poster_url.strip() != "" and "placeholder.com" not in poster_url:
#                 continue

#             key = next(api_keys)
#             poster = fetch_poster(name, key)

#             cursor.execute(
#                 "UPDATE movies SET poster_url = ? WHERE movie_id = ?",
#                 (poster, movie_id)
#             )

#             # Optional: small delay to avoid hammering API
#             time.sleep(0.2)

#         conn.commit()
#         print(f"✅ Batch {offset} → {offset + len(batch)} updated")

#     conn.close()
#     print("🎉 All batches processed!")

# # === 5. Run ===
# if __name__ == "__main__":
#     update_posters()
import sqlite3
import requests
import time
import os

# -----------------------------
# Config
# -----------------------------
OMDB_API_KEY = "507dbc7c"
OMDB_URL = "https://www.omdbapi.com/"
DB_PATH = "instance/movies.db"  # path to your existing init.db

# -----------------------------
# Update posters function
# -----------------------------
def update_posters(start_row=5001, batch_size=500):
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get total rows in table
    cursor.execute("SELECT COUNT(*) FROM movies")
    total_rows = cursor.fetchone()[0]

    print(f"Total movies in DB: {total_rows}")

    # Fetch all movies from start_row to the last row
    cursor.execute(
        "SELECT rowid AS movie_id, name FROM movies WHERE rowid >= ? ORDER BY rowid",
        (start_row,)
    )
    rows = cursor.fetchall()
    total = len(rows)
    print(f"📦 {total} movies from row {start_row} to {total_rows}. Starting poster updates...")

    start = 0
    batch_num = 1
    while start < total:
        batch = rows[start:start + batch_size]
        if not batch:
            break
        print(f"⏳ Updating batch {batch_num} ({len(batch)} movies)")
        for i, row in enumerate(batch):
            mid, name = row
            try:
                params = {"t": name, "apikey": OMDB_API_KEY}
                r = requests.get(OMDB_URL, params=params, timeout=5)
                data = r.json()
                poster = data.get("Poster")
                if not poster or poster == "N/A":
                    poster = f"https://via.placeholder.com/300x420/1a1a2e/ffffff?text={name.replace(' ', '+')}"
                cursor.execute(
                    "UPDATE movies SET poster_url = ? WHERE rowid = ?",
                    (poster, mid)
                )
                conn.commit()
                print(f"✅ [{i+1}/{len(batch)}] Poster updated for '{name}'")
            except Exception as e:
                print(f"❌ [{i+1}/{len(batch)}] Error updating '{name}': {e}")
            time.sleep(0.8)  # small delay to avoid hammering API
        start += batch_size
        batch_num += 1

    conn.close()
    print(f"🎉 Posters for movies from row {start_row} to {total_rows} updated successfully!")

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    update_posters(start_row=5001, batch_size=500)
