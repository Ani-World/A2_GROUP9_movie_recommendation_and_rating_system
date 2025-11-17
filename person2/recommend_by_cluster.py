import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

DATA_PATH = r"C:\Users\Admin\OneDrive\Desktop\College work\AIML FINAL\AIML-Mini-project\person2\models\genre_name_mapping.csv"

df = pd.read_csv(DATA_PATH)

# Combine Genre text with cluster id for richer feature
df["GenreText"] = df["Genre"].fillna("").astype(str) + " cluster" + df["GenreCluster"].astype(str)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["GenreText"])
cos_sim = cosine_similarity(tfidf_matrix)

def recommend_by_genre(movie_query, top_n=5):
    names = df["Name"].astype(str)
    lowered = names.str.lower()
    q = movie_query.strip().lower()

    matches = df[lowered.str.contains(q, na=False)]
    if matches.empty:
        close = get_close_matches(q, lowered, n=1, cutoff=0.6)
        if close:
            matches = df[lowered == close[0]]
    if matches.empty:
        print(f"No movies found matching '{movie_query}'")
        return None, []

    idx = matches.index[0]
    movie_name = df.loc[idx, "Name"]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i for i, _ in sim_scores[1 : top_n + 15]]
    same_cluster = df.loc[idx, "GenreCluster"]
    filtered = df.iloc[top_indices]
    filtered = filtered[filtered["GenreCluster"] == same_cluster]
    recs = filtered["Name"].unique().tolist()[:top_n]

    return movie_name, recs

if __name__ == "__main__":
    q = input("Enter a movie name (partial allowed): ").strip()
    try:
        n = int(input("How many recommendations do you want? (default 5): ").strip() or 5)
    except:
        n = 5

    exact, recs = recommend_by_genre(q, top_n=n)
    if recs:
        print(f"\n🎬 Recommendations similar to '{exact}':")
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r}")
