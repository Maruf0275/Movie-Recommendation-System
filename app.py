"""Streamlit UI for CineMatch movie recommendations."""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource
def load_bundle():
    path = Path(__file__).with_name("movie_recommender.pkl")
    with path.open("rb") as f:
        return pickle.load(f)  # Only open models you trained or otherwise trust.


bundle = load_bundle()
movies = bundle["movies"]

st.title("🎬 CineMatch")
st.write("Find movies similar to your favorites, using genre and community-tag features.")
st.caption("Content-based recommendations, not personalized ratings or watch-history predictions.")

with st.form("recommend_form"):
    title = st.selectbox("Choose a movie", sorted(movies["title"].tolist()))
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
    with col2:
        min_ratings = st.slider("Minimum number of ratings", min_value=0, max_value=50, value=0, step=5)
    submitted = st.form_submit_button("Find similar movies", use_container_width=True, type="primary")

if submitted:
    selected = movies[movies["title"].eq(title)].iloc[0]
    query = bundle["vectorizer"].transform([selected["content"]])
    count = min(len(movies), max(250, top_n + 1))
    distances, indices = bundle["model"].kneighbors(query, n_neighbors=count)
    choices = movies.iloc[indices[0]].copy()
    choices["Similarity (%)"] = ((1 - distances[0]) * 100).clip(0, 100).round(1)
    choices = choices[(choices["movieId"] != selected["movieId"]) & (choices["rating_count"] >= min_ratings)]
    choices = choices.sort_values(["Similarity (%)", "rating_count"], ascending=[False, False]).head(top_n)
    st.subheader(f"Movies similar to {title}")
    if choices.empty:
        st.warning("No matching recommendations. Reduce the minimum ratings filter.")
    else:
        shown = choices.rename(columns={"title": "Movie", "genres": "Genres", "average_rating": "Average rating", "rating_count": "Rating count"})
        st.dataframe(shown[["Movie", "Genres", "Average rating", "Rating count", "Similarity (%)"]], use_container_width=True, hide_index=True)
        st.caption("Similarity is a cosine similarity score, not a probability that you will enjoy the movie.")

st.divider()
st.subheader("Dataset")
c1, c2 = st.columns(2)
c1.metric("Movies available", f"{len(movies):,}")
c2.metric("Recommendation method", "TF-IDF + KNN")
st.caption("MovieLens Latest Small, GroupLens Research. Dataset subject to its usage license; not for commercial use without permission.")
