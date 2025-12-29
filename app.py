import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Recommender System",
    layout="wide"
)

# ---------------- TMDB API ----------------
API_KEY = "8919f825912033cfb30789a1d80dcd2b"

# ---------------- LOAD & PROCESS DATA ----------------
@st.cache_data
def load_and_prepare_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")

    movies = movies[[
        "movie_id", "title", "overview",
        "genres", "keywords", "cast", "crew"
    ]]

    movies.dropna(inplace=True)

    def convert(text):
        return [i["name"] for i in eval(text)]

    def get_director(text):
        for i in eval(text):
            if i["job"] == "Director":
                return [i["name"]]
        return []

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(lambda x: convert(x)[:3])
    movies["crew"] = movies["crew"].apply(get_director)

    for col in ["genres", "keywords", "cast", "crew"]:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    new_df = movies[["movie_id", "title", "tags"]]
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()

    similarity = cosine_similarity(vectors)

    return new_df, similarity

movies, similarity = load_and_prepare_data()

# ---------------- FETCH POSTER ----------------
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {
            "api_key": API_KEY,
            "language": "en-US"
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        poster_path = data.get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster"

    except:
        return "https://via.placeholder.com/500x750?text=API+Error"

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters

# ---------------- UI ----------------
st.title("🎬 Movie Recommender System")
st.markdown("Content-Based Movie Recommendation using Cosine Similarity")

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)

if st.button("Show Recommendations"):
    names, posters = recommend(selected_movie)

    cols = st.columns(5)
    for col, name, poster in zip(cols, names, posters):
        with col:
            st.text(name)
            st.image(poster, use_container_width=True)
