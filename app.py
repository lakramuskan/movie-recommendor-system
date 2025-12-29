import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Recommender",
    layout="wide",
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(180deg, #0b0f2f 0%, #060814 100%);
}
.main {
    background: linear-gradient(180deg, #0b0f2f 0%, #060814 100%);
}
.hero-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: white;
}
.hero-sub {
    font-size: 20px;
    text-align: center;
    color: #cfcfcf;
    margin-bottom: 30px;
}
.movie-card {
    background: #0f172a;
    border-radius: 14px;
    padding: 10px;
    transition: transform 0.3s ease;
}
.movie-card:hover {
    transform: scale(1.05);
}
.movie-title {
    color: white;
    font-size: 14px;
    margin-top: 8px;
    text-align: center;
}
.section-title {
    color: white;
    font-size: 26px;
    font-weight: 700;
    margin: 30px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TMDB API ----------------
API_KEY = "8919f825912033cfb30789a1d80dcd2b"

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": API_KEY, "language": "en-US"}
        data = requests.get(url, params=params).json()
        if data.get("poster_path"):
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    except:
        pass
    return "https://via.placeholder.com/500x750?text=No+Image"

# ---------------- LOAD & BUILD MODEL ----------------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")

    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
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

movies, similarity = load_data()

# ---------------- RECOMMEND ----------------
def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]

    names, posters = [], []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))
    return names, posters

# ---------------- HERO SECTION ----------------
st.markdown('<div class="hero-title">Find Movies You’ll Enjoy<br>Without The Hassle.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Search through thousands of movies</div>', unsafe_allow_html=True)

selected_movie = st.selectbox(
    "",
    movies["title"].values,
    label_visibility="collapsed"
)

# ---------------- RECOMMENDATIONS ----------------
if st.button("🎬 Recommend"):
    names, posters = recommend(selected_movie)

    st.markdown('<div class="section-title">Recommended For You</div>', unsafe_allow_html=True)
    cols = st.columns(5)

    for col, name, poster in zip(cols, names, posters):
        with col:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            st.image(poster, use_container_width=True)
            st.markdown(f'<div class="movie-title">{name}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ALL MOVIES GRID ----------------
st.markdown('<div class="section-title">All Movies</div>', unsafe_allow_html=True)

grid_cols = st.columns(5)
for idx, row in movies.sample(15).iterrows():
    with grid_cols[idx % 5]:
        poster = fetch_poster(row.movie_id)
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        st.image(poster, use_container_width=True)
        st.markdown(f'<div class="movie-title">{row.title}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
