# backend/app.py
"""
Updated single-file Flask backend with idempotent /api/onboarding (dedupe),
and the usual recommendation/predict/movie endpoints (stubs for ML models).
Replace model stubs with real models later — the code prints model names when called.

Endpoints:
- POST /api/auth/register
- POST /api/auth/login
- GET  /api/movies/onboarding
- POST /api/onboarding   <-- includes dedupe protection (returns cached response if duplicate within window)
- POST /api/rate
- GET  /api/recommendations?user_id=UID&n=20
- GET  /api/predict?user_id=UID&movie_id=MID
- GET  /api/movie/<movie_id>
- GET  /api/health
"""

import os, time, hashlib, json
from collections import defaultdict
from math import log1p
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import numpy as np
import joblib
import requests


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Load KMeans model
kmeans_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))

# Load Regressor model
regressor_model = joblib.load(os.path.join(MODEL_DIR, "regressor_model.pkl"))

print(f"Loaded models: {type(kmeans_model).__name__}, {type(regressor_model).__name__}")

from flask import send_from_directory

app = Flask(__name__)
CORS(app)

# -----------------------------
# Database configuration
# -----------------------------
db_path = os.path.join(os.path.dirname(__file__), '..', 'instance', 'movies.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

import threading, time, requests

# -----------------------------
# Load movies quickly
# -----------------------------
_movies_map = {}

def load_movies_quickly():
    global _movies_map
    with app.app_context():
        try:
            with db.engine.connect() as conn:
                result = conn.execute(db.text("SELECT rowid AS movie_id, * FROM movies"))
                for row in result.mappings():
                    mid = int(row.get("movie_id"))
                    name = row.get("name") or "Unknown"
                    poster_url = row.get("poster_url") or f"https://via.placeholder.com/300x420/1a1a2e/ffffff?text={name.replace(' ', '+')}"

                    _movies_map[mid] = {
                        "movie_id": mid,
                        "Name": name,
                        "Year": int(row.get("year") or 0),
                        "Duration": int(row.get("duration") or 0),
                        "Rating": float(row.get("avg_rating") or 0.0),
                        "Votes": int(row.get("votes") or 0),
                        "Director": row.get("director") or "Unknown",
                        "Actor 1": row.get("actor1") or "Unknown",
                        "Actor 2": row.get("actor2") or "Unknown",
                        "Actor 3": row.get("actor3") or "Unknown",
                        "poster": poster_url,
                        "popularity": int(row.get("votes") or 100),
                    }

            print(f"✅ Loaded {len(_movies_map)} movies from movies.db (startup fast mode)")
        except Exception as e:
            print("❌ Error loading movies from database:", e)

load_movies_quickly()
# -----------------------------
# Variables requested (names used)
# -----------------------------
genre_cols = ["Genre_Action", "Genre_Drama", "Genre_Romance"]
# Build movie_features (genre_vector + normalized Year/Duration/Rating/log(Votes))
movie_features_list = []
for movie_id in sorted(_movies_map.keys()):
    v = [1 if movie_id % (i + 2) == 0 else 0 for i in range(len(genre_cols))]
    genre_vector = np.array(v, dtype=float)
    norm_year = (_movies_map[movie_id]["Year"] - 2000) / 25.0
    norm_duration = _movies_map[movie_id]["Duration"] / 150.0
    norm_rating = _movies_map[movie_id]["Rating"] / 5.0
    log_votes = np.log1p(_movies_map[movie_id]["Votes"])
    fv = np.concatenate([genre_vector, [norm_year, norm_duration, norm_rating, log_votes]])
    movie_features_list.append(fv)
movie_features = np.vstack(movie_features_list)

cluster_assignments = {mid: int((mid - 1) % 4) for mid in _movies_map.keys()}
rating_matrix = defaultdict(dict)   # user_id -> {movie_id: rating}
user_profile = defaultdict(lambda: np.zeros(movie_features.shape[1]))

# Model stubs
class KMeansStub:
    name = "KMeansStub"
    def predict(self, X): return np.array([int(i % 4) for i in range(len(X))])
kmeans_model = KMeansStub()

class RegressorStub:
    name = "RegressorStub"
    def predict(self, X):
        base = np.array([_movies_map[m]["Rating"] for m in sorted(_movies_map.keys())])
        mean = float(base.mean())
        return np.clip(np.ones((X.shape[0],)) * mean, 0.0, 5.0)
regressor_model = RegressorStub()

apriori_rules = {12: ["Because users who liked 3 also liked 12."], 8: ["Often watched with 3."]}

hybrid_weights = {"regression": 0.5, "knn": 0.3, "apriori": 0.2}

# -----------------------------
# Dedupe storage for /api/onboarding
# -----------------------------
_last_onboarding_submission = {}   # user_id -> (payload_hash, timestamp, last_response)
_ONBOARDING_DEDUPE_WINDOW = 8      # seconds

# -----------------------------
# Helper functions
# -----------------------------
def movie_to_output(m):
    return {
        "movie_id": m["movie_id"],
        "name": m.get("Name", "Unknown"),
        "year": int(m.get("Year", 0)),
        "duration": int(m.get("Duration", 0)),
        "rating": float(m.get("Rating", 0.0)),
        "votes": int(m.get("Votes", 0)),
        "director": m.get("Director", "Unknown"),
        "actor1": m.get("Actor 1", "Unknown"),
        "actor2": m.get("Actor 2", "Unknown"),
        "actor3": m.get("Actor 3", "Unknown"),
        "poster": m.get("poster", f"https://via.placeholder.com/300x420?text={m['movie_id']}"),
        "popularity_score": round(
            (m.get("popularity", 100) * (m.get("Rating", 0) / 5.0) * log1p(m.get("Votes", 1))), 3
        )
    }

def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0: return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_user_profile(user_id):
    liked = [mid for mid, r in rating_matrix[user_id].items() if r >= 4]
    if not liked: return np.zeros(movie_features.shape[1])
    idxs = [sorted(_movies_map.keys()).index(mid) for mid in liked if mid in _movies_map]
    if not idxs: return np.zeros(movie_features.shape[1])
    return np.mean(movie_features[idxs, :], axis=0)

def get_knn_scores_for_user(user_id):
    up = get_user_profile(user_id)
    scores = {}
    keys = sorted(_movies_map.keys())
    for i, mid in enumerate(keys):
        mv = movie_features[i]
        scores[mid] = cosine_sim(up, mv)
    return scores

def get_regressor_preds_for_user(user_id, mids):
    up = get_user_profile(user_id)
    X = []
    keys = sorted(_movies_map.keys())
    for mid in mids:
        idx = keys.index(mid)
        mv = movie_features[idx]
        X.append(np.concatenate([up, mv]))
    X = np.vstack(X) if X else np.zeros((0, movie_features.shape[1] * 2))
    preds = regressor_model.predict(X)
    return {mid: float(preds[i]) for i, mid in enumerate(mids)}

def apriori_boost_for_user(user_id, mids):
    boosts = {mid: 0.0 for mid in mids}
    reasons = defaultdict(list)
    liked_mids = [mid for mid, r in rating_matrix[user_id].items() if r >= 4]
    for rec_mid in mids:
        rules = apriori_rules.get(rec_mid, [])
        for rule in rules:
            for lm in liked_mids:
                if str(lm) in rule or str(lm) in rule:
                    boosts[rec_mid] += 1.0
                    reasons[rec_mid].append(rule)
    maxb = max(boosts.values()) if boosts else 0.0
    if maxb > 0:
        for k in boosts: boosts[k] = boosts[k] / maxb
    return boosts, reasons

# -----------------------------
# Simple auth (in-memory) for demo
# -----------------------------
_users = {}
_next_user_id = 1



@app.route('/api/auth/register', methods=['POST'])
def register():
    global _next_user_id
    data = request.get_json(force=True) or {}
    email = (data.get('email') or '').strip().lower()
    pw = data.get('password') or ''
    name = data.get('name') or ''
    if not email or not pw: return jsonify({"error": "email and password required"}), 400
    if email in _users: return jsonify({"error": "user already exists"}), 409
    uid = _next_user_id
    _next_user_id += 1
    _users[email] = {"user_id": uid, "email": email, "name": name, "password": pw}
    return jsonify({"message": "registered", "user_id": uid}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json(force=True) or {}
    email = (data.get('email') or '').strip().lower()
    pw = data.get('password') or ''
    if not email or not pw: return jsonify({"error": "email and password required"}), 400
    user = _users.get(email)
    if not user or user.get("password") != pw: return jsonify({"error": "invalid credentials"}), 401
    return jsonify({"message": "ok", "user_id": user["user_id"]}), 200

# -----------------------------
# Movies for onboarding
# -----------------------------
import random

@app.route('/api/movies/onboarding', methods=['GET'])
def movies_onboarding():
    # Filter: only movies with avg_rating > 3
    eligible = [m for m in _movies_map.values() if m.get("Rating", 0) > 3.8 and m.get("Votes", 0) >= 500]

    # Randomly select 25 (or fewer if not enough)
    random_movies = random.sample(eligible, min(25, len(eligible)))

    # Format the output
    movies = [
        {
            "movie_id": m["movie_id"],
            "name": m["Name"],
            "year": m["Year"],
            "poster": m.get("poster"),
            "rating": m.get("Rating"),
        }
        for m in random_movies
    ]

    return jsonify({"movies": movies}), 200

# -----------------------------
# Onboarding with dedupe
# -----------------------------
@app.route('/api/onboarding', methods=['POST'])
def onboarding():
    global _last_onboarding_submission
    data = request.get_json(force=True) or {}
    user_id = data.get('user_id')
    responses = data.get('responses') or []

    if not user_id or not isinstance(responses, list):
        return jsonify({"error": "user_id and responses required"}), 400

    # Build deterministic payload hash
    try:
        normalized = {"user_id": int(user_id), "responses": sorted(
            [{"movie_id": int(r.get("movie_id")), "like": int(r.get("like"))} for r in responses],
            key=lambda x: (x["movie_id"], x["like"])
        )}
        payload_bytes = json.dumps(normalized, separators=(',', ':'), sort_keys=True).encode('utf-8')
    except Exception:
        payload_bytes = json.dumps(data, separators=(',', ':'), sort_keys=True).encode('utf-8')

    payload_hash = hashlib.sha1(payload_bytes).hexdigest()
    now = time.time()

    # Dedupe check (avoid reprocessing identical submissions)
    last = _last_onboarding_submission.get(user_id)
    if last:
        last_hash, last_ts, last_resp = last
        if payload_hash == last_hash and (now - last_ts) <= _ONBOARDING_DEDUPE_WINDOW:
            app.logger.info(f"onboarding: duplicate submission detected for user {user_id}, returning cached response")
            return jsonify(last_resp), 200

    # Process and save ratings into rating_matrix
    saved = 0
    for r in responses:
        try:
            movie_id = int(r.get('movie_id'))
            like_val = int(r.get('like'))
            if like_val not in (-1, 0, 1):
                continue
        except Exception:
            continue
        # Map like -> pseudo-rating scale (e.g., dislike=1, neutral=3, like=5)
        pseudo_rating = 5.0 if like_val == 1 else (3.0 if like_val == 0 else 1.0)
        rating_matrix[user_id][movie_id] = pseudo_rating
        saved += 1

    # ---------------------------------------------
    # Improved Recommendations (cluster + popularity)
    # ---------------------------------------------
    liked_movie_ids = [r["movie_id"] for r in responses if r.get("like") == 1]
    if liked_movie_ids:
        liked_clusters = [cluster_assignments.get(mid) for mid in liked_movie_ids if mid in cluster_assignments]
        if liked_clusters:
            # Most common cluster among liked movies
            top_cluster = max(set(liked_clusters), key=liked_clusters.count)
            recs = [m for m in _movies_map.values() if cluster_assignments.get(m["movie_id"]) == top_cluster]
            recs = sorted(recs, key=lambda x: x["popularity"], reverse=True)[:10]
        else:
            # Fallback to popularity if cluster not found
            recs = sorted(_movies_map.values(), key=lambda x: x["popularity"], reverse=True)[:10]
    else:
        # Fallback if no liked movies
        recs = sorted(_movies_map.values(), key=lambda x: x["popularity"], reverse=True)[:10]

    # Format output same as before
    out = [
        {
            "movie_id": m["movie_id"],
            "name": m["Name"],
            "year": m["Year"],
            "poster": m.get("poster"),
        }
        for m in recs
    ]

    response_body = {"message": f"saved {saved} responses", "recommendations": out}

    # Cache the last onboarding submission for dedupe
    _last_onboarding_submission[user_id] = (payload_hash, now, response_body)
    app.logger.info(f"onboarding: saved {saved} responses for user {user_id}")

    return jsonify(response_body), 200


# -----------------------------
# Rate endpoint (for testing)
# -----------------------------
@app.route('/api/rate', methods=['POST'])
def api_rate():
    data = request.get_json(force=True) or {}
    user_id = data.get('user_id')
    movie_id = data.get('movie_id')
    rating = data.get('rating')
    rating = rating / 2.0

    if user_id is None or movie_id is None or rating is None:
        return jsonify({"error":"user_id, movie_id, rating required"}), 400
    try:
        user_id = int(user_id); movie_id = int(movie_id); rating = float(rating)
    except:
        return jsonify({"error":"invalid types"}), 400
    rating_matrix[user_id][movie_id] = rating
    return jsonify({"message":"rating saved"}), 201

# -----------------------------
# Recommendations (hybrid)
# -----------------------------
@app.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    user_id = request.args.get('user_id')
    n = int(request.args.get('n', 20))
    if user_id is None:
        return jsonify({"error":"user_id required"}), 400
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error":"user_id must be int-like"}), 400

    # Print model names (stubs) so logs show which to replace later
    app.logger.info("Calling models: %s %s %s", kmeans_model.name, type(apriori_rules).__name__, regressor_model.name)

    rated = set(rating_matrix[user_id].keys())
    candidate_mids = [mid for mid in sorted(_movies_map.keys()) if mid not in rated]
    if not candidate_mids:
        return jsonify({"recommendations": []}), 200

    reg_preds = get_regressor_preds_for_user(user_id, candidate_mids)
    knn_scores = get_knn_scores_for_user(user_id)
    apr_boosts, apr_reasons = apriori_boost_for_user(user_id, candidate_mids)

    items = []
    for mid in candidate_mids:
        reg_score = reg_preds.get(mid, 0.0) / 5.0
        knn_score = knn_scores.get(mid, 0.0)
        apr_score = apr_boosts.get(mid, 0.0)
        hybrid_score = (hybrid_weights["regression"] * reg_score +
                        hybrid_weights["knn"] * knn_score +
                        hybrid_weights["apriori"] * apr_score)
        m = _movies_map[mid]
        out = movie_to_output(m)
        out["cluster_id"] = cluster_assignments.get(mid)
        out["hybrid_score"] = round(float(hybrid_score), 4)
        out["reg_score"] = round(float(reg_score),4)
        out["rating"] = round(float(out["rating"]), 1) if "rating" in out else 0.0
        out["predicted_display"] = round(float(reg_score) * 10 / 5, 1)
        out["knn_score"] = round(float(knn_score),4)
        out["apr_score"] = round(float(apr_score),4)
        out["apriori_reasons"] = apr_reasons.get(mid, [])

        items.append(out)

    items_sorted = sorted(items, key=lambda x: (x["hybrid_score"], x["popularity_score"]), reverse=True)
    return jsonify({"items": items_sorted[:n]}), 200

# -----------------------------
# Predict
# -----------------------------
@app.route('/api/predict', methods=['GET'])
def api_predict():
    user_id = request.args.get('user_id')
    movie_id = request.args.get('movie_id')
    if user_id is None or movie_id is None:
        return jsonify({"error":"user_id and movie_id required"}), 400
    try:
        user_id = int(user_id); movie_id = int(movie_id)
    except:
        return jsonify({"error":"user_id and movie_id must be ints"}), 400
    if movie_id not in _movies_map:
        return jsonify({"error":"movie_id not found"}), 404

    app.logger.info("Predict called with models: %s %s", regressor_model.name, kmeans_model.name)
    pred = get_regressor_preds_for_user(user_id, [movie_id]).get(movie_id, _movies_map[movie_id]["Rating"])
    num_user_ratings = len(rating_matrix[user_id])
    confidence = min(0.95, 0.2 + 0.05 * num_user_ratings)
    return jsonify({
    "name": _movies_map[movie_id]["Name"],
    "movie_id": movie_id,
    "predicted_rating": round(float(pred) * 2, 1),  
    "confidence": round(float(confidence),3),
    "source": "hybrid_regressor_knn_stub"
}), 200


# -----------------------------
# Movie metadata
# -----------------------------
@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def api_movie(movie_id):
    OMDB_API_KEY = "2f0ad043"
    OMDB_URL = "https://www.omdbapi.com/"

    movie = _movies_map.get(movie_id)
    if not movie:
        return jsonify({"error": "Movie not found"}), 404

    # If poster is missing or placeholder, fetch on-demand
    poster_url = movie.get("poster")
    if (
        not poster_url
        or "placeholder.com" in poster_url
        or poster_url.strip() == ""
        or poster_url.strip().upper() == "N/A"
    ):
        try:
            params = {"t": movie["Name"], "apikey": OMDB_API_KEY}
            r = requests.get(OMDB_URL, params=params, timeout=3)
            data = r.json()

            if data.get("Poster") and data["Poster"] != "N/A":
                poster_url = data["Poster"]
                movie["poster"] = poster_url
                # Save it in DB safely
                with db.engine.begin() as conn:
                    conn.execute(
                        db.text("UPDATE movies SET poster_url = :p WHERE rowid = :id"),
                        {"p": poster_url, "id": movie_id},
                    )
                print(f"✅ Poster fetched on demand for '{movie['Name']}'")
            else:
                poster_url = f"https://via.placeholder.com/300x420/1a1a2e/ffffff?text={movie['Name'].replace(' ', '+')}"
                movie["poster"] = poster_url
                print(f"⚠️ No poster found for '{movie['Name']}', using placeholder.")
        except Exception as e:
            print(f"⚠️ Poster fetch failed for '{movie['Name']}': {e}")
            poster_url = f"https://via.placeholder.com/300x420/1a1a2e/ffffff?text={movie['Name'].replace(' ', '+')}"
            movie["poster"] = poster_url

    # Return movie details with valid poster
    return jsonify({
        "movie_id": movie_id,
        "Name": movie["Name"],
        "year": movie["Year"],
        "duration": movie["Duration"],
        "Rating": movie["Rating"],
        "Votes": movie["Votes"],
        "poster": movie["poster"],
        "Director": movie["Director"],
        "Actor 1": movie["Actor 1"],
        "Actor 2": movie["Actor 2"],
        "Actor 3": movie["Actor 3"],
    })

# -----------------------------
# Archive (user history) - simple read
# -----------------------------
@app.route('/api/user/archive', methods=['GET'])
def api_archive():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error":"user_id required"}), 400
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error":"user_id must be int-like"}), 400
    rated = rating_matrix[user_id]
    out = []
    for mid, r in rated.items():
        m = _movies_map.get(mid)
        if m:
            o = movie_to_output(m)
            o["user_rating"] = r
            out.append(o)
    return jsonify({"archived": out}), 200

# -----------------------------
# Health
# -----------------------------
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status":"ok"}), 200
from flask import send_from_directory

# -----------------------------
# Frontend routes (absolute path)
# -----------------------------
FRONTEND_DIR = r"C:\Users\Admin\OneDrive\Desktop\AIML-Mini-project\templates"  # <-- paste your actual path here

@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/dashboard')
def serve_dashboard():
    return send_from_directory(FRONTEND_DIR, 'Dashboard.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(FRONTEND_DIR, path)

# -----------------------------
# Run
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.logger.info("Starting backend on port %d", port)
    app.run(host='127.0.0.1', port=port, debug=True)
