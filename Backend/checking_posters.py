

import os
import requests
from flask import Flask, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

app = Flask(__name__)

# -----------------------------
# Database config
# -----------------------------
db_path = os.path.join(os.path.dirname(__file__), '..', 'instance', 'movies.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# -----------------------------
# Helper function
# -----------------------------
def movie_to_output(m):
    poster_url = m.get("poster_url")
    
    # Check if URL actually works; fallback to placeholder
    try:
        if poster_url:
            r = requests.head(poster_url, timeout=3)
            if r.status_code != 200:
                poster_url = None
        else:
            poster_url = None
    except:
        poster_url = None

    return {
        "movie_id": m["movie_id"],
        "name": m.get("name", "Unknown"),
        "poster": poster_url or f"https://via.placeholder.com/150x210?text={m.get('name','Unknown').replace(' ', '+')}"
    }

# -----------------------------
# Route to fetch 20 movies from row 750-800
# -----------------------------
@app.route('/movies')
def show_movies():
    query = text("""
        SELECT rowid AS movie_id, name, poster_url
        FROM movies
        WHERE rowid BETWEEN 200 AND 420
        ORDER BY RANDOM()
        LIMIT 20
    """)
    with db.engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    movies = [movie_to_output(r) for r in rows]
    return jsonify(movies)

# -----------------------------
# Frontend route to display 20 movies from row 750-800
# -----------------------------
@app.route('/')
def index():
    query = text("""
        SELECT rowid AS movie_id, name, poster_url
        FROM movies
        WHERE rowid BETWEEN 200 AND 420
        ORDER BY RANDOM()
        LIMIT 20
    """)
    with db.engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    movies = [movie_to_output(r) for r in rows]

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Movies 750-800</title>
        <style>
            body { font-family: Arial; background: #f4f4f4; }
            .container { display: flex; flex-wrap: wrap; gap: 20px; padding: 20px; }
            .movie { width: 150px; text-align: center; background: #fff; padding: 10px; border-radius: 8px; box-shadow: 0 0 5px #ccc; }
            img { width: 150px; height: 210px; object-fit: cover; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Movies 750-800</h1>
        <div class="container">
            {% for m in movies %}
            <div class="movie">
                <img src="{{ m.poster }}" alt="{{ m.name }}">
                <h4>{{ m.name }}</h4>
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """
    return render_template_string(html, movies=movies)

# -----------------------------
# Run server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
