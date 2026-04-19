from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
IMAGE_FOLDER = "images"
MOVIE_FILE = "Short_data.csv"
SIMILARITY_FILE = "similarity.pkl"

# Load initial data
movies = pd.read_csv(MOVIE_FILE)
similarity = pickle.load(open(SIMILARITY_FILE, "rb"))

@app.route("/")
def home():
    images = [(img, os.path.splitext(img)[0]) for img in os.listdir(IMAGE_FOLDER) if img.endswith((".jpg", ".png", ".jpeg"))]
    return render_template("index.html", images=images)

@app.route("/images/<path:filename>")
def get_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route('/submit_selection', methods=['POST'])
def submit_selection():
    data = request.get_json()
    selected_movies = data.get("selected_movies", [])
    return jsonify({"selected_movies": selected_movies})

# Recommendation function
def recommend(selected_movies):
    recommended_movies = set()
    for movie_name in selected_movies:
        if movie_name in movies["title"].values:  
            index = movies[movies["title"] == movie_name].index[0]              
            similar_movies = sorted(
                list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True
            )[1:6]            
            for i in similar_movies:
                recommended_movies.add(movies.iloc[i[0]]["title"])  # Extract title
    return list(recommended_movies)

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    selected_movies = data.get("selected_movies", [])
    recommendations = recommend(selected_movies)
    return jsonify({"recommended_movies": recommendations})

@app.route("/add_movie", methods=["GET", "POST"])
def add_movies():
    if request.method == "POST":
        title = request.form.get("title")
        overview = request.form.get("overview")
        genre = request.form.get("genre")
        keywords = request.form.get("keywords")
        cast = request.form.get("cast")
        crew = request.form.get("crew")
        
        new_movie = pd.DataFrame({
            "title": [title],
            "overview": [overview],
            "genres": [genre],
            "keywords": [keywords],
            "cast": [cast],
            "crew": [crew]
        })
        
        new_movie.to_csv(MOVIE_FILE, mode='a', header=False, index=False)
        retrain_model()
        return jsonify({"message": "Movie added and model retrained successfully!"})
    
    return render_template("add_movie.html")

# Retraining function
def retrain_model():
    global movies, similarity
    movies = pd.read_csv(MOVIE_FILE)
    movies.fillna("", inplace=True)
    movies["combined_features"] = movies["overview"] + " " + movies["genres"] + " " + movies["keywords"] + " " + movies["cast"] + " " + movies["crew"]
    movies["combined_features"].apply(lambda x:" ".join(x))
    movies["combined_features"].apply(lambda x:x.lower())
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_matrix = vectorizer.fit_transform(movies["combined_features"]).toarray()
    similarity = cosine_similarity(feature_matrix)
    with open(SIMILARITY_FILE, "wb") as file:
        pickle.dump(similarity, file)

if __name__ == "__main__":
    app.run(debug=True)









#python3 recommender.py