'''
File : recommend_by_content.py
Works: This code provides content-based movie recommendations using cosine similarity on genres: it asks the user for the dataset path and movie name, finds the most similar movies based on genre overlap, and saves both a CSV and a bar plot of the recommended movies in the plots folder.
'''


import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Ask user for dataset path
DATA_PATH = input("Enter the full path of your preprocessed movies CSV: ")

# Check if file exists
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"No file found at '{DATA_PATH}'")

# Load data
df = pd.read_csv(DATA_PATH)

# Select only genre columns (boolean columns)
genre_cols = [col for col in df.columns if col.startswith('Genre_')]
df_genres = df[genre_cols].astype(int)  # Ensure numeric

# Compute cosine similarity between movies
cos_sim = cosine_similarity(df_genres)

# Function to get top N recommendations
def recommend(movie_name, top_n=5):
    if movie_name not in df['Name'].values:
        print(f"Movie '{movie_name}' not found in dataset.")
        return
    idx = df.index[df['Name'] == movie_name][0]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]  # skip the movie itself
    recommendations = df.iloc[top_indices][['Name', 'Year', 'Rating']]
    return recommendations

# Ask user for movie input
movie_input = input("Enter the movie name for recommendations: ")
recs = recommend(movie_input)

if recs is not None:
    # Ensure plots folder exists
    os.makedirs('plots', exist_ok=True)
    
    # Save recommendations to CSV
    csv_path = os.path.join('plots', f'recommendations_for_{movie_input}.csv')
    recs.to_csv(csv_path, index=False)
    print(f"Recommendations for '{movie_input}' saved in {csv_path}.")

    # Save a simple bar plot of Ratings
    plt.figure(figsize=(8,4))
    plt.barh(recs['Name'], recs['Rating'], color='skyblue')
    plt.xlabel('Rating')
    plt.title(f'Top {len(recs)} Recommendations for "{movie_input}"')
    plt.gca().invert_yaxis()
    png_path = os.path.join('plots', f'recommendations_for_{movie_input}.png')
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    print(f"Bar plot saved at {png_path}.")
