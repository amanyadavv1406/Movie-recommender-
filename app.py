
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('movies.csv')
df['description'] = df['description'].fillna('')

# Vectorize descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Mapping titles to indices
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Recommendation function
def recommend(title, num_recommendations=5):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Streamlit UI
st.title("Movie Recommender App")
st.write("Enter a movie title and get similar movies!")

movie_title = st.text_input("Movie Title", "")

if movie_title:
    results = recommend(movie_title)
    if results:
        st.subheader(f"Movies similar to '{movie_title}':")
        for movie in results:
            st.write(f"- {movie}")
    else:
        st.error("Movie not found. Please check the title.")
