import os, time, hashlib, json, logging
from collections import defaultdict
from math import log1p
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import numpy as np
import joblib
import requests
import pandas as pd

# --- Configuration and Setup ---
# Configure logging for better output visibility
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

# NOTE: Using a static path for the database file location.
# It is assumed that 'movies.db' is located in the 'instance' folder 
# next to the 'backend' folder.
db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'instance'))
db_path = os.path.join(db_dir, 'movies.db')

# Ensure the instance directory exists for the SQLite file
os.makedirs(db_dir, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'a_secure_secret_key_for_session_management'

db = SQLAlchemy(app)

# --- Load All Models, Scalers, and Metadata ---

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

try:
    # Load Models
    kmeans_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    regressor_model = joblib.load(os.path.join(MODEL_DIR, "regressor_model.pkl"))

    # Load the Scaler for the regressor
    scaler_reg = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

    # Load the metadata file
    meta_path = os.path.join(MODEL_DIR, "feature_cols.json")
    with open(meta_path, 'r') as f:
        model_metadata = json.load(f)

    # Get the list of columns the regressor model was trained on
    regression_feature_cols = model_metadata["regression_feature_cols"]

    print(f"✅ Loaded models: {type(kmeans_model).__name__}, {type(regressor_model).__name__}")
    print(f"✅ Loaded scaler: {type(scaler_reg).__name__}")
    print(f"✅ Loaded {len(regression_feature_cols)} regression features from metadata.")

except Exception as e:
    print(f"❌ CRITICAL ERROR loading models: {e}. Check the 'Backend/models' folder.")
    # Stop the app if models can't be loaded
    raise
# --- Load Apriori Rules (or initialize empty) ---
apriori_rules = {} # Default to empty dict in case file is missing
try:
    apriori_path = os.path.join(MODEL_DIR, "apriori_rules.json")
    if os.path.exists(apriori_path):
        with open(apriori_path, 'r') as f:
            apriori_rules = json.load(f)
        print(f"✅ Loaded {len(apriori_rules)} apriori rules from JSON.")
    else:
        # This is not an error, it just means this feature will be disabled
        print("⚠️ 'apriori_rules.json' not found. Apriori boosts will be 0.")
except Exception as e:
    print(f"❌ Error loading 'apriori_rules.json': {e}. Using empty rules.")

# --- SQLAlchemy Database Models ---

class User(db.Model):
    # This table stores persistent user authentication data
    __tablename__ = 'users' 
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(64))

class Rating(db.Model):
    # This table corresponds to your existing 'ratings' table structure
    __tablename__ = 'ratings'
    id = db.Column(db.Integer, primary_key=True)
    # Note: user_id and movie_id are now foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', name='fk_ratings_user_id'), nullable=False) 
    movie_id = db.Column(db.Integer, nullable=False) # Refers to movie table ROWID
    rating = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.Float, default=time.time)

class Watchlist(db.Model):
    __tablename__ = 'watchlist'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', name='fk_watchlist_user_id'), nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.Float, default=time.time)
    # Add a unique constraint to prevent adding the same movie twice
    __table_args__ = (db.UniqueConstraint('user_id', 'movie_id', name='_user_movie_uc'),)

# --- Initial Data Loading (Movies and Ratings) ---

_movies_map = {}
rating_matrix = defaultdict(dict) # user_id -> {movie_id: rating}
_ONBOARDING_DEDUPE_WINDOW = 8 # seconds
_last_onboarding_submission = {}
genre_cols=[]

# Simple password hashing helper (using sha256)
def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def load_movies_quickly():
    global _movies_map
    try:
        with db.engine.connect() as conn:
            # Note: Using db.text() is safer and necessary when selecting rowid 
            result = conn.execute(db.text("SELECT rowid AS movie_id, * FROM movies"))
            
            # Map column names to lowercase/consistent keys for compatibility
            column_names = result.keys()
            
            for row in result:
                row_dict = dict(zip(column_names, row))
                mid = int(row_dict.get("movie_id"))
                name = row_dict.get("name") or "Unknown"
                poster_url = row_dict.get("poster_url") or f"https://via.placeholder.com/300x420/1a1a2e/ffffff?text={name.replace(' ', '+')}"

                _movies_map[mid] = {
                    "movie_id": mid,
                    "Name": name,
                    "Year": int(row_dict.get("year") or 0),
                    "Duration": int(row_dict.get("duration") or 0),
                    "Rating": float(row_dict.get("avg_rating") or 0.0),
                    "Votes": int(row_dict.get("votes") or 0),
                    "Director": row_dict.get("director") or "Unknown",
                    "Actor 1": row_dict.get("actor1") or "Unknown",
                    "Actor 2": row_dict.get("actor2") or "Unknown",
                    "Actor 3": row_dict.get("actor3") or "Unknown",
                    "poster": poster_url,
                    "popularity": int(row_dict.get("votes") or 100),
                }

        print(f"✅ Loaded {len(_movies_map)} movies from movies.db")
    except Exception as e:
        print(f"❌ Error loading movies from database: {e}")

def load_all_ratings_from_db():
    global rating_matrix # Make sure to modify the global variable
    # Clear it first in case of reload
    rating_matrix = defaultdict(dict)
    try:
        # Load all ratings from the persistent table into the in-memory cache
        ratings_data = db.session.execute(db.select(Rating)).scalars().all()
        count = 0
        for r in ratings_data:
            rating_matrix[r.user_id][r.movie_id] = r.rating
            count += 1
        print(f"✅ Loaded {count} existing ratings from database into cache.")
    except Exception as e:
        print(f"❌ Error loading existing ratings: {e}")



# Call the initial loading functions
with app.app_context():
    load_movies_quickly()

# --- Feature/Model Setup (Using data from name_features_mapping.csv) ---

movie_features_list = []
movie_id_map_index = {}
cluster_assignments = {}  # This will be filled with REAL cluster IDs
sorted_movie_ids = sorted(_movies_map.keys())

# Define the path to your mapping data file
mapping_path = os.path.join(MODEL_DIR, 'name_features_mapping.csv')

try:
    # Load the mapping data CSV
    df_data = pd.read_csv(mapping_path).drop_duplicates(subset=['Name'], keep='first')
    # Create a fast lookup map using the movie Name
    df_data_map = df_data.set_index("Name")
    print(f"✅ Loaded '{mapping_path}' for feature and cluster generation.")

    # Create a default zero-vector for fallback
    zero_vector = np.zeros(len(regression_feature_cols))

    for i, movie_id in enumerate(sorted_movie_ids):
        movie_id_map_index[movie_id] = i
        m = _movies_map.get(movie_id) # Get movie from DB map
        if not m:
            continue
            
        movie_name = m.get("Name")

        try:
            # Find the movie's feature row from the mapping CSV
            feature_row = df_data_map.loc[movie_name]
            
            # 1. Extract the REAL feature vector in the correct order
            feature_vector_list = [feature_row.get(col, 0) for col in regression_feature_cols]
            fv = np.array(feature_vector_list, dtype=float)
            
            # 2. Extract the REAL cluster ID
            cluster_id = int(feature_row.get("Cluster", -1))
            cluster_assignments[movie_id] = cluster_id
            
        except KeyError:
            # Movie from DB not found in CSV, use a zero-vector and default cluster
            fv = zero_vector
            cluster_assignments[movie_id] = -1 # Assign to a default 'unknown' cluster
        except Exception as e:
            print(f"Error processing {movie_name}: {e}")
            fv = zero_vector
            cluster_assignments[movie_id] = -1
            
        movie_features_list.append(fv)

    # Create the final, correct movie_features array
    if movie_features_list:
        movie_features = np.vstack(movie_features_list)
        print(f"✅ Built movie_features array with REAL data. Shape: {movie_features.shape}")
        print(f"✅ Built REAL cluster_assignments map for {len(cluster_assignments)} movies.")
    else:
        movie_features = np.zeros((0, len(regression_feature_cols)))
        print("⚠️ Built EMPTY movie_features array.")

except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load or process 'name_features_mapping.csv'. {e}")
    # We can't continue if this fails, so re-raise
    raise

# (Keep rating_matrix, user_profile definitions)
#rating_matrix = defaultdict(dict)   # user_id -> {movie_id: rating}
user_profile = defaultdict(lambda: np.zeros(movie_features.shape[1]))


#cluster_assignments = {mid: int((mid - 1) % 4) for mid in _movies_map.keys()}
hybrid_weights = {"regression": 0.5, "knn": 0.3, "apriori": 0.2}

# --- Helper Functions (No Change) ---
def movie_to_output(m):
    # ... (Keep existing movie_to_output logic)
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
    idxs = [movie_id_map_index.get(mid) for mid in liked if mid in movie_id_map_index]
    idxs = [idx for idx in idxs if idx is not None]
    if not idxs: return np.zeros(movie_features.shape[1])
    return np.mean(movie_features[idxs, :], axis=0)

def get_knn_scores_for_user(user_id):
    up = get_user_profile(user_id)
    scores = {}
    for mid, i in movie_id_map_index.items():
        mv = movie_features[i]
        scores[mid] = cosine_sim(up, mv)
    return scores

def get_regressor_preds_for_user(user_id, mids):
    
    # 1. Get the global mean values from our loaded "scaler" dict
    # --- FIX: Use 'scaler_reg' instead of 'target_encoder_means' ---
    g_mean = scaler_reg['global_mean_rating']
    g_mean_genre = scaler_reg['g_mean_rat_global']
    genre_means_list = scaler_reg['genre_means_list']
    dir_means = scaler_reg['dir_mean_rat']
    a1_means = scaler_reg['a1_mean_rat']
    a2_means = scaler_reg['a2_mean_rat']
    a3_means = scaler_reg['a3_mean_rat']
    # genre_cols is already defined globally
    # --- END FIX ---

    features_list = []
    valid_mids = []
    
    for mid in mids:
        if mid not in _movies_map:
            continue
            
        m = _movies_map[mid]
        
        # 2. Build the feature vector for this one movie
        features = {}
        features['Year'] = m.get('Year', g_mean)
        features['Votes'] = m.get('Votes', 0)
        features['Duration'] = m.get('Duration', 100)
        
        # 3. Apply the learned target encoding means
        features['Dir_enc'] = dir_means.get(m.get('Director'), g_mean)
        features['A1_enc'] = a1_means.get(m.get('Actor 1'), g_mean)
        features['A2_enc'] = a2_means.get(m.get('Actor 2'), g_mean)
        features['A3_enc'] = a3_means.get(m.get('Actor 3'), g_mean)
        
        # 4. Calculate this movie's weighted genre score
        g_score = 0
        g_count = 0
        idx = movie_id_map_index.get(mid)
        if idx is not None:
            # movie_features[idx] is the content vector (Genres + Year, Votes, Duration)
            # The first part (length of genre_cols) is the genre vector
            movie_genre_vector = movie_features[idx][:len(genre_cols)]
            
            for i, val in enumerate(movie_genre_vector):
                if val == 1:
                    # Use the pre-calculated mean for this genre
                    g_score += genre_means_list[i]
                    g_count += 1
        
        features['G_mean_rat'] = (g_score / g_count) if g_count > 0 else g_mean_genre
            
        # 5. Ensure final vector is in the correct order
        final_vector = [features[col] for col in regression_feature_cols]
        features_list.append(final_vector)
        valid_mids.append(mid)

    if not valid_mids:
        return {}

    # Convert to DataFrame to ensure correct shape/types
    X_predict = pd.DataFrame(features_list, columns=regression_feature_cols)
    
    # 6. Predict using the new features
    preds = regressor_model.predict(X_predict)
    
    return {mid: float(preds[i]) for i, mid in enumerate(valid_mids)}

def apriori_boost_for_user(user_id, mids):
    boosts = {mid: 0.0 for mid in mids}
    reasons = defaultdict(list)
    liked_mids = [mid for mid, r in rating_matrix[user_id].items() if r >= 4]
    # NOTE: Keep the stub logic simple as it depends on external apriori_rules structure
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
# 🔄 AUTH ENDPOINTS (Rewritten for DB Persistence) 🔄
# -----------------------------

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json(force=True) or {}
    email = (data.get('email') or '').strip().lower()
    pw = data.get('password') or ''
    name = data.get('name') or ''
    
    if not email or not pw: 
        return jsonify({"error": "email and password required"}), 400
    
    with app.app_context():
        # Check if user already exists
        existing_user = db.session.execute(db.select(User).filter_by(email=email)).scalar_one_or_none()
        if existing_user:
            return jsonify({"error": "user already exists"}), 409
        
        # Create new user and save to database
        password_hash = hash_password(pw)
        new_user = User(email=email, password_hash=password_hash, name=name)
        
        db.session.add(new_user)
        db.session.commit()
        
        # Reload cache to ensure new user exists (though no ratings yet)
        user_id = new_user.id
        
    return jsonify({"message": "registered", "user_id": user_id}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json(force=True) or {}
    email = (data.get('email') or '').strip().lower()
    pw = data.get('password') or ''
    
    if not email or not pw: 
        return jsonify({"error": "email and password required"}), 400
    
    with app.app_context():
        # Retrieve user by email
        user = db.session.execute(db.select(User).filter_by(email=email)).scalar_one_or_none()
        
        if not user or user.password_hash != hash_password(pw): 
            return jsonify({"error": "invalid credentials"}), 401
            
        user_id = user.id
        
    return jsonify({"message": "ok", "user_id": user_id}), 200

# -----------------------------
# Movies for onboarding (No Change)
# -----------------------------
import random

@app.route('/api/movies/onboarding', methods=['GET'])
def movies_onboarding():
    # ... (Keep existing movies_onboarding logic)
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
# 🔄 Onboarding (Rewritten for DB Persistence) 🔄
# -----------------------------
@app.route('/api/onboarding', methods=['POST'])
def onboarding():
    global _last_onboarding_submission
    data = request.get_json(force=True) or {}
    user_id = data.get('user_id')
    responses = data.get('responses') or []

    if not user_id or not isinstance(responses, list):
        return jsonify({"error": "user_id and responses required"}), 400

    # ... (Dedupe hash logic - KEEP)
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

    # Process and save ratings into database and cache
    saved = 0
    new_ratings = []
    
    with app.app_context():
        # Clear previous onboarding ratings for this user in case of re-submission
        db.session.execute(db.delete(Rating).where(Rating.user_id == user_id))
        
        # Clear in-memory cache for fresh start
        if int(user_id) in rating_matrix:
            rating_matrix[int(user_id)] = {}

        for r in responses:
            try:
                movie_id = int(r.get('movie_id'))
                like_val = int(r.get('like'))
                if like_val not in (-1, 0, 1): continue
            except Exception: continue
            
            # Map like -> pseudo-rating scale (1.0, 3.0, 5.0)
            pseudo_rating = 5.0 if like_val == 1 else (3.0 if like_val == 0 else 1.0)
            
            # 1. Save to Database
            new_ratings.append(Rating(user_id=user_id, movie_id=movie_id, rating=pseudo_rating))
            
            # 2. Update In-memory Cache
            rating_matrix[int(user_id)][movie_id] = pseudo_rating
            saved += 1
            
        db.session.add_all(new_ratings)
        db.session.commit()

    # ... (Keep existing recommendation generation logic for response)
    liked_movie_ids = [r["movie_id"] for r in responses if r.get("like") == 1]
    if liked_movie_ids:
        liked_clusters = [cluster_assignments.get(mid) for mid in liked_movie_ids if mid in cluster_assignments]
        if liked_clusters:
            top_cluster = max(set(liked_clusters), key=liked_clusters.count)
            recs = [m for m in _movies_map.values() if cluster_assignments.get(m["movie_id"]) == top_cluster]
            recs = sorted(recs, key=lambda x: x["popularity"], reverse=True)[:10]
        else:
            recs = sorted(_movies_map.values(), key=lambda x: x["popularity"], reverse=True)[:10]
    else:
        recs = sorted(_movies_map.values(), key=lambda x: x["popularity"], reverse=True)[:10]

    out = [
        {"movie_id": m["movie_id"], "name": m["Name"], "year": m["Year"], "poster": m.get("poster")}
        for m in recs
    ]

    response_body = {"message": f"saved {saved} responses", "recommendations": out}
    _last_onboarding_submission[user_id] = (payload_hash, now, response_body)
    app.logger.info(f"onboarding: saved {saved} responses persistently for user {user_id}")
    return jsonify(response_body), 200

# -----------------------------
# 🔄 Rate endpoint (Rewritten for DB Persistence) 🔄
# -----------------------------
@app.route('/api/rate', methods=['POST'])
def api_rate():
    data = request.get_json(force=True) or {}
    user_id = data.get('user_id')
    movie_id = data.get('movie_id')
    rating = data.get('rating') # Assuming this is 1-10 from frontend
    
    # Scale 1-10 (frontend) to 1-5 (model/DB)
    if rating is not None:
        rating_db = float(rating) / 2.0
    else:
        rating_db = None

    if user_id is None or movie_id is None or rating_db is None:
        return jsonify({"error":"user_id, movie_id, rating required"}), 400
    try:
        user_id = int(user_id); movie_id = int(movie_id); rating_db = float(rating_db)
    except:
        return jsonify({"error":"invalid types"}), 400
    
    with app.app_context():
        # Check if rating exists (for update)
        existing_rating = db.session.execute(
            db.select(Rating).filter_by(user_id=user_id, movie_id=movie_id)
        ).scalar_one_or_none()

        if existing_rating:
            # Update existing rating
            existing_rating.rating = rating_db
            existing_rating.timestamp = time.time()
        else:
            # Create new rating
            new_rating = Rating(user_id=user_id, movie_id=movie_id, rating=rating_db)
            db.session.add(new_rating)
            
        db.session.commit()
    
    # Update in-memory cache after DB commit
    rating_matrix[user_id][movie_id] = rating_db
    
    return jsonify({"message": f"rating {rating_db} saved persistently"}), 201

# -----------------------------
# Remaining Endpoints (Logic based on rating_matrix remains the same)
# -----------------------------

@app.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    # ... (Keep existing api_recommendations logic, which uses in-memory rating_matrix)
    user_id = request.args.get('user_id')
    n = int(request.args.get('n', 20))
    if user_id is None:
        return jsonify({"error":"user_id required"}), 400
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error":"user_id must be int-like"}), 400
    
    app.logger.info("Calling models: %s %s %s", type(kmeans_model).__name__, type(apriori_rules).__name__, type(regressor_model).__name__)

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
        out["rating"] = round(float(out.get("rating", 0.0)), 1)
        out["predicted_display"] = round(float(reg_score) * 10 / 5, 1)
        out["knn_score"] = round(float(knn_score),4)
        out["apr_score"] = round(float(apr_score),4)
        out["apriori_reasons"] = apr_reasons.get(mid, [])

        items.append(out)

    items_sorted = sorted(items, key=lambda x: (x["hybrid_score"], x["popularity_score"]), reverse=True)
    return jsonify({"items": items_sorted[:n]}), 200

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

    app.logger.info("Predict called with models: %s %s", type(regressor_model).__name__, type(kmeans_model).__name__)
    
    # 1. Get the base prediction from the model
    pred = get_regressor_preds_for_user(user_id, [movie_id]).get(movie_id, _movies_map[movie_id]["Rating"])
    
    # 2. Get the base confidence from number of ratings
    num_user_ratings = len(rating_matrix[user_id])
    confidence = min(0.95, 0.2 + 0.05 * num_user_ratings)
    
    # --- START FIX: Boost confidence with Apriori ---
    try:
        # 3. Check for Apriori boosts
        apr_boosts, _ = apriori_boost_for_user(user_id, [movie_id])
        if apr_boosts.get(movie_id, 0.0) > 0:
            # If a rule was found, add a confidence bonus (e.g., +0.25)
            confidence = min(0.99, confidence + 0.25) # Cap at 0.99
            app.logger.info(f"Confidence boost applied for movie {movie_id} via Apriori rule.")
    except Exception as e:
        app.logger.error(f"Error applying apriori boost to confidence: {e}")
    # --- END FIX ---

    return jsonify({
        "name": _movies_map[movie_id]["Name"],
        "movie_id": movie_id,
        "predicted_rating": round(float(pred), 1), 
        "confidence": round(float(confidence),3),
        "source": "hybrid_regressor_knn_stub"
    }), 200


@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def api_movie(movie_id):
    # ... (Keep existing api_movie logic, including OMDB fetch and DB update for poster)
    OMDB_API_KEY = "2f0ad043"
    OMDB_URL = "https://www.omdbapi.com/"

    movie = _movies_map.get(movie_id)
    if not movie:
        return jsonify({"error": "Movie not found"}), 404

    # Poster fetch logic (uses OMDB, updates DB, and updates in-memory map)
    poster_url = movie.get("poster")
    if (not poster_url or "placeholder.com" in poster_url or poster_url.strip() == "" or poster_url.strip().upper() == "N/A"):
        try:
            params = {"t": movie["Name"], "apikey": OMDB_API_KEY}
            r = requests.get(OMDB_URL, params=params, timeout=3)
            data = r.json()

            if data.get("Poster") and data["Poster"] != "N/A":
                poster_url = data["Poster"]
                movie["poster"] = poster_url
                with db.engine.begin() as conn: # Uses raw DB engine for simple update
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

@app.route('/api/user/archive', methods=['GET'])
def api_archive():
    # ... (Keep existing api_archive logic)
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error":"user_id required"}), 400
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error":"user_id must be int-like"}), 400
    rated = rating_matrix[user_id] # Reads from the now persistently loaded cache
    out = []
    for mid, r in rated.items():
        m = _movies_map.get(mid)
        if m:
            o = movie_to_output(m)
            o["user_rating"] = r 
            out.append(o)
    return jsonify({"archived": out}), 200

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status":"ok", "movies_loaded": len(_movies_map)}), 200


@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error": "user_id must be int-like"}), 400

    with app.app_context():
        items = db.session.execute(
            db.select(Watchlist).filter_by(user_id=user_id).order_by(Watchlist.timestamp.desc())
        ).scalars().all()
        
        movie_ids = [item.movie_id for item in items]
        
    # Get movie details from our in-memory map
    watchlist_movies = []
    for mid in movie_ids:
        if mid in _movies_map:
            watchlist_movies.append(movie_to_output(_movies_map[mid]))
            
    return jsonify({"watchlist": watchlist_movies}), 200

@app.route('/api/watchlist/add', methods=['POST'])
def add_to_watchlist():
    data = request.get_json(force=True) or {}
    user_id = data.get('user_id')
    movie_id = data.get('movie_id')

    if not user_id or not movie_id:
        return jsonify({"error": "user_id and movie_id required"}), 400
    
    try:
        with app.app_context():
            # Check if it's already on the list
            existing = db.session.execute(
                db.select(Watchlist).filter_by(user_id=user_id, movie_id=movie_id)
            ).scalar_one_or_none()
            
            if existing:
                return jsonify({"message": "Movie already on watchlist"}), 200
                
            # Add new item
            new_item = Watchlist(user_id=user_id, movie_id=movie_id)
            db.session.add(new_item)
            db.session.commit()
            
        return jsonify({"message": "Movie added to watchlist"}), 201
        
    except Exception as e:
        # Handle potential unique constraint violation
        db.session.rollback()
        if "UNIQUE constraint failed" in str(e):
             return jsonify({"message": "Movie already on watchlist"}), 200
        app.logger.error(f"Error adding to watchlist: {e}")
        return jsonify({"error": "Could not add to watchlist"}), 500

# -----------------------------
# Frontend routes (Absolute path)
# -----------------------------
# NOTE: The path provided by the user must be correct for these to work.
FRONTEND_DIR =r"C:\Users\DELL\Projects\AIML-Mini-Project-semifinal\templates"

@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/dashboard')
def serve_dashboard():
    return send_from_directory(FRONTEND_DIR, 'Dashboard.html')

@app.route('/<path:path>')
def serve_static_files(path):
    # Fallback to serve any other file (like JS, CSS, or other HTML pages)
    return send_from_directory(FRONTEND_DIR, path)

# -----------------------------
# Run
# -----------------------------
if __name__ == '__main__':
    with app.app_context():
        # 1. Create User and Rating tables if they don't exist
        db.create_all()
        # 2. Load all persistent user ratings into the in-memory cache
        load_all_ratings_from_db()
        
    port = int(os.environ.get("PORT", 5000))
    app.logger.info("Starting backend on port %d", port)
    app.run(host='127.0.0.1', port=port, debug=True)