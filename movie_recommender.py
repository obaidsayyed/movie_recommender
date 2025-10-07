import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the aggregated dataset (one row per movie, with average rating)
data = pd.read_csv(r"C:\Users\Rohit Sarkar\OneDrive\Desktop\Obaid Codec\final-model-dataset.csv")

# Keep relevant columns (assuming 'rating' column exists with average ratings)
data = data[['title', 'genres', 'year', 'rating']].drop_duplicates().reset_index(drop=True)
data['genres'] = data['genres'].replace("(no genres listed)", "")
data['title_lower'] = data['title'].str.lower()

# Initialize CountVectorizer and fit on all genres
cv = CountVectorizer(token_pattern='[^|]+')
genre_matrix = cv.fit_transform(data['genres'])

# Recommendation function with formatted output
def recommend(movie_title, top_n=5, top_similar_n=20):
    """
    Recommend movies based on genre similarity and average rating.
    
    top_n: number of movies to return
    top_similar_n: number of most similar movies to consider before sorting by rating
    """
    movie_title = movie_title.strip().lower()
    
    if movie_title not in list(data['title_lower']):
        return f"Movie '{movie_title}' not found in the dataset. Please check spelling."
    
    # Index of the input movie
    idx = data[data['title_lower'] == movie_title].index[0]

    # Compute similarity of input movie with all movies
    sim_scores = cosine_similarity(genre_matrix[idx], genre_matrix).flatten()

    # Ignore the input movie itself
    sim_scores[idx] = -1

    # Get top N most similar movies by genre
    top_indices = sim_scores.argsort()[::-1][:top_similar_n]
    top_similar = data.iloc[top_indices].copy()

    # Sort these top similar movies by rating
    top_similar_sorted = top_similar.sort_values(by='rating', ascending=False).head(top_n)

    # Format output as "Title (Year) — Rating"
    top_similar_sorted['Recommendation'] = top_similar_sorted.apply(
        lambda row: f"{row['title']} ({int(row['year'])}) — {row['rating']}", axis=1
    )

    return top_similar_sorted[['Recommendation']]

# User input
movie_name = input("Enter a movie name: ")
recommended_movies = recommend(movie_name, top_n=5)

print("\nTop 5 recommended movies based on your choice:\n")
print(recommended_movies.to_string(index=False))
