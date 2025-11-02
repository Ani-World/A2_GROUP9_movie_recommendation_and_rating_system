import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load clean dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "datasets", "movies_data_processed.csv")
df = pd.read_csv(data_path)

# Example feature matrix
X = df[["Year", "Duration", "Rating", "Votes"]].fillna(0)

# Train KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
joblib.dump(kmeans, "Backend/models/kmeans_model.pkl")

# Train Linear Regressor
regressor = LinearRegression()
regressor.fit(X, df["Rating"])
joblib.dump(regressor, "Backend/models/regressor_model.pkl")

print("✅ Models saved: kmeans_model.pkl, regressor_model.pkl")
