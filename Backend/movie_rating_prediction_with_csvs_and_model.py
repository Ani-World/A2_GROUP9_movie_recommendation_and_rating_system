# movie_rating_prediction_with_csvs_and_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("datasets\movies_data.csv")
print("Data Info:")
print(df.info())
print("\nDescriptive Stats:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# 2. Data Cleaning & Preprocessing
# -----------------------------
df['Duration'] = df['Duration'].replace(0, np.nan)
for col in ['Genre', 'Actor 1', 'Actor 2', 'Actor 3']:
    df['Duration'] = df.groupby(col)['Duration'].transform(lambda x: x.fillna(x.median()))
df = df[df['Duration'] >= 60]

# Split Genre to multiple rows if multiple genres exist
df['Genre'] = df['Genre'].fillna('Unknown')
df = df.assign(Genre=df['Genre'].str.split(',')).explode('Genre')
df['Genre'] = df['Genre'].str.strip()
df.dropna(subset=['Rating', 'Actor 1', 'Actor 2', 'Actor 3'], inplace=True)

# -----------------------------
# 3. Feature Engineering & Save Encodings
# -----------------------------
def encode_mean_rating_save(col_name, df, folder="encodings"):
    os.makedirs(folder, exist_ok=True)
    mean_series = df.groupby(col_name)['Rating'].mean()
    mean_csv_path = os.path.join(folder, f"{col_name}_mean_rating.csv")
    mean_series.to_csv(mean_csv_path, header=['MeanRating'])
    print(f"Saved {col_name} mean-rating CSV: {mean_csv_path}")
    return df[col_name].map(mean_series)

df['G_mean_rat'] = encode_mean_rating_save('Genre', df)
df['Dir_enc'] = encode_mean_rating_save('Director', df)
df['A1_enc'] = encode_mean_rating_save('Actor 1', df)
df['A2_enc'] = encode_mean_rating_save('Actor 2', df)
df['A3_enc'] = encode_mean_rating_save('Actor 3', df)

# Drop original categorical columns
df.drop(columns=['Name','Genre','Director','Actor 1','Actor 2','Actor 3'], inplace=True)

# Features & target
X = df.drop(columns=['Rating'])
y = df['Rating']

# -----------------------------
# 4. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 5. Model Training & Evaluation
# -----------------------------
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "DecisionTreeRegressor_max14": DecisionTreeRegressor(max_depth=14)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    
    print(f"\nModel: {name}")
    print(f" MSE: {mse:.3f}")
    print(f" MAE: {mae:.3f}")
    print(f" R2: {r2:.3f}")
    print(f" 5-Fold CV R2 Mean: {cv_scores.mean():.3f}, Std: {cv_scores.std():.3f}")

# -----------------------------
# 6. Save Decision Tree Model
# -----------------------------
dt_model = DecisionTreeRegressor(max_depth=14)
dt_model.fit(X_train, y_train)
os.makedirs("saved_models", exist_ok=True)
model_path = os.path.join("saved_models", "decision_tree_regressor.pkl")
joblib.dump(dt_model, model_path)
print(f"\n Decision Tree Regressor saved at: {model_path}")
