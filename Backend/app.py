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
from flask_cors import CORS
import numpy as np
import joblib
import os

# Path to models folder
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Load KMeans model
kmeans_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))

# Load Regressor model
regressor_model = joblib.load(os.path.join(MODEL_DIR, "regressor_model.pkl"))

print(f"Loaded models: {type(kmeans_model).__name__}, {type(regressor_model).__name__}")

app = Flask(__name__)
CORS(app)

# -----------------------------
# Simple in-memory / lightweight persistence (for demo)
# -----------------------------
# Seed 20 movies (movie_id 1..20) with CSV-like fields
_movies = [
    (1, "The First Dawn", 2018, 110, 4.1, 1200, "Dir A", "Actor A1", "Actor A2", "Actor A3", 120),
    (2, "Silent Echoes", 2019, 95, 3.9, 950, "Dir B", "Actor B1", "Actor B2", "Actor B3", 95),
    (3, "Midnight Run", 2020, 125, 4.5, 2000, "Dir C", "Actor C1", "Actor C2", "Actor C3", 200),
    (4, "River of Stars", 2017, 105, 4.2, 1500, "Dir D", "Actor D1", "Actor D2", "Actor D3", 150),
    (5, "Lonely Planet", 2016, 100, 3.7, 800, "Dir E", "Actor E1", "Actor E2", "Actor E3", 80),
    (6, "Neon Street", 2021, 115, 4.3, 1400, "Dir F", "Actor F1", "Actor F2", "Actor F3", 140),
    (7, "Fading Letters", 2015, 98, 3.5, 650, "Dir G", "Actor G1", "Actor G2", "Actor G3", 65),
    (8, "Glass Garden", 2022, 130, 4.6, 1800, "Dir H", "Actor H1", "Actor H2", "Actor H3", 180),
    (9, "Paper Wings", 2014, 90, 3.2, 550, "Dir I", "Actor I1", "Actor I2", "Actor I3", 55),
    (10, "Stone Harbor", 2013, 108, 3.8, 700, "Dir J", "Actor J1", "Actor J2", "Actor J3", 70),
    (11, "Echo Valley", 2021, 112, 4.0, 1100, "Dir K", "Actor K1", "Actor K2", "Actor K3", 110),
    (12, "Crimson Tide", 2020, 122, 4.4, 1600, "Dir L", "Actor L1", "Actor L2", "Actor L3", 160),
    (13, "Hidden Paths", 2019, 99, 3.9, 900, "Dir M", "Actor M1", "Actor M2", "Actor M3", 90),
    (14, "Velvet Morning", 2018, 104, 4.15, 1300, "Dir N", "Actor N1", "Actor N2", "Actor N3", 130),
    (15, "Paper Moon", 2017, 97, 3.6, 750, "Dir O", "Actor O1", "Actor O2", "Actor O3", 75),
    (16, "Neptune's Call", 2016, 103, 3.85, 820, "Dir P", "Actor P1", "Actor P2", "Actor P3", 85),
    (17, "Wandering Light", 2015, 92, 3.3, 600, "Dir Q", "Actor Q1", "Actor Q2", "Actor Q3", 60),
    (18, "City of Quiet", 2014, 94, 3.1, 500, "Dir R", "Actor R1", "Actor R2", "Actor R3", 50),
    (19, "Last Harbor", 2013, 106, 3.05, 420, "Dir S", "Actor S1", "Actor S2", "Actor S3", 40),
    (20, "Aurora Lane", 2022, 118, 4.55, 1700, "Dir T", "Actor T1", "Actor T2", "Actor T3", 170),
]

_movies_map = {}
for t in _movies:
    movie_id = t[0]
    _movies_map[movie_id] = {
        "movie_id": movie_id,
        "Name": t[1],
        "Year": t[2],
        "Duration": t[3],
        "Rating": float(t[4]),
        "Votes": int(t[5]),
        "Director": t[6],
        "Actor 1": t[7],
        "Actor 2": t[8],
        "Actor 3": t[9],
        "poster": f"https://via.placeholder.com/300x420?text={movie_id}",
        "popularity": t[10],
    }

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
        "Name": m["Name"],
        "Year": int(m["Year"]),
        "Duration": int(m["Duration"]),
        "avg_rating": float(m["Rating"]),
        "votes": int(m["Votes"]),
        "director": m["Director"],
        "actor1": m["Actor 1"],
        "actor2": m["Actor 2"],
        "actor3": m["Actor 3"],
        "poster": m.get("poster"),
        "popularity_score": round(m["popularity"] * (m["Rating"] / 5.0) * log1p(m["Votes"]), 3)
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
@app.route('/api/movies/onboarding', methods=['GET'])
def movies_onboarding():
    movies = [ {"movie_id": m["movie_id"], "name": m["Name"], "year": m["Year"], "poster": m.get("poster")} for m in _movies_map.values() ]
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

    # Build simple recommendations: movies not rated by user ordered by popularity
    rated_movie_ids = set(rating_matrix[user_id].keys())
    recs = [m for m in sorted(_movies_map.values(), key=lambda x: x["popularity"], reverse=True) if m["movie_id"] not in rated_movie_ids][:10]
    out = [{"movie_id": m["movie_id"], "name": m["Name"], "year": m["Year"], "poster": m.get("poster")} for m in recs]

    response_body = {"message": f"saved {saved} responses", "recommendations": out}

    # store dedupe cache
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
        out["knn_score"] = round(float(knn_score),4)
        out["apr_score"] = round(float(apr_score),4)
        out["apriori_reasons"] = apr_reasons.get(mid, [])
        items.append(out)

    items_sorted = sorted(items, key=lambda x: (x["hybrid_score"], x["popularity_score"]), reverse=True)
    return jsonify({"recommendations": items_sorted[:n]}), 200

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
        "Name": _movies_map[movie_id]["Name"],
        "movie_id": movie_id,
        "predicted_rating": round(float(pred),3),
        "confidence": round(float(confidence),3),
        "source": "hybrid_regressor_knn_stub"
    }), 200

# -----------------------------
# Movie metadata
# -----------------------------
@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def api_movie(movie_id):
    if movie_id not in _movies_map:
        return jsonify({"error":"movie not found"}), 404
    m = _movies_map[movie_id]
    out = movie_to_output(m)
    out["cluster_id"] = cluster_assignments.get(movie_id)
    out["apriori_reasons"] = apriori_rules.get(movie_id, [])
    app.logger.info("Movie endpoint used models: %s apriori:%s", kmeans_model.name, type(apriori_rules).__name__)
    return jsonify(out), 200

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
    return jsonify({"archive": out}), 200

# -----------------------------
# Health
# -----------------------------
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status":"ok"}), 200

# -----------------------------
# Run
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.logger.info("Starting backend on port %d", port)
    app.run(host='127.0.0.1', port=port, debug=True)
