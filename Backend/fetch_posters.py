import os
import requests
import time
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

# --- Configuration ---
# This is the API key you already have in your app.py
OMDB_API_KEY = "ENTER API KEY HERE" 
OMDB_URL = "https://www.omdbapi.com/"

# --- Database Setup (copied from your app.py) ---
db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'instance'))
db_path = os.path.join(db_dir, 'movies.db')

if not os.path.exists(db_path):
    print(f"❌ Database file not found at {db_path}")
    print("Please make sure your database exists before running this script.")
    exit()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Helper Function ---
def fetch_poster_url(movie_name):
    """
    Fetches a single movie poster URL from OMDB.
    """
    try:
        params = {"t": movie_name, "apikey": OMDB_API_KEY}
        r = requests.get(OMDB_URL, params=params, timeout=5) # 5 second timeout
        r.raise_for_status() # Raise an error for bad responses (4xx, 5xx)
        data = r.json()
        
        if data.get("Response") == "True" and data.get("Poster") and data["Poster"] != "N/A":
            return data["Poster"]
        else:
            print(f"  -> No poster found on OMDB for '{movie_name}'.")
            return f"https://via.placeholder.com/300x420/1a1a2e/ffffff?text={movie_name.replace(' ', '+')}"
    except requests.exceptions.Timeout:
        print(f"  -> Timeout while fetching '{movie_name}'.")
    except Exception as e:
        print(f"  -> Network error for '{movie_name}': {e}")
        
    # Fallback placeholder
    return f"https://via.placeholder.com/300x420/1a1a2e/ffffff?text={movie_name.replace(' ', '+')}"

# --- Main Execution ---
def run_prefetch():
    with app.app_context():
        # 1. Find all movies that need a poster update
        query = text("""
            SELECT rowid, name, poster_url FROM movies 
            WHERE poster_url IS NULL 
               OR poster_url = 'N/A' 
               OR poster_url = ''
               OR poster_url LIKE '%placeholder.com%'
        """)
        
        with db.engine.connect() as conn:
            movies_to_fetch = conn.execute(query).mappings().all()
        
        total = len(movies_to_fetch)
        print(f"Found {total} movies that need poster pre-fetching.")
        
        if total == 0:
            print("Database is already up to date. Exiting.")
            return

        # 2. Loop, fetch, and update (with a rate limit)
        for i, movie in enumerate(movies_to_fetch):
            movie_id = movie['rowid']
            movie_name = movie['name']
            
            print(f"[{i+1}/{total}] Fetching poster for: {movie_name}...")
            
            new_poster_url = fetch_poster_url(movie_name)
            
            # 3. Update the database
            with db.engine.begin() as conn: # Auto-commits
                conn.execute(
                    text("UPDATE movies SET poster_url = :url WHERE rowid = :id"),
                    {"url": new_poster_url, "id": movie_id}
                )
            
            print(f"  -> Saved: {new_poster_url[:60]}...")
            
            # 4. IMPORTANT: Rate limit to be kind to the free API
            # Do not remove this!
            time.sleep(0.5) # 2 requests per second

        print("\n✅ Poster pre-fetch complete!")

if __name__ == "__main__":
    run_prefetch()