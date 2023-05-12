import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template, flash
import warnings
warnings.filterwarnings("ignore")
import json

app = Flask(__name__)
app.config.update(
    TESTING = True,
    SECRET_KEY = "password"
)

ratings = pd.read_csv('Dataset/ratings.csv')
books = pd.read_csv('Dataset/books_cleaned.csv')

with open("model.pkl", 'rb') as f:
    svd_model = pickle.load(f)

def get_book_id(book_title):
    # Returns books id for given book title
    book = books[books['title'] == book_title]
    if len(book) > 0:
        return book['book_id'].iloc[0]
    else:
        return None

def book_read(user_id):
    '''Take user_id and return list of book that user has read'''
    books_list = list(books['book_id'])
    book_read_list = list(ratings['book_id'][ratings['user_id'] == user_id])
    return books_list, book_read_list

def get_new_user_id(title_ratings):
    # Get book IDs for the given book titles
    book_ids = [get_book_id(title) for title in title_ratings.keys()]
    
    # Create a new row for the new user
    new_user_id = ratings['user_id'].max() + 1
    new_user_ratings = pd.DataFrame({
        'user_id': [new_user_id] * len(book_ids),
        'book_id': book_ids,
        'rating': list(title_ratings.values())
    })
    
    # Append the new user's ratings to the existing ratings dataframe
    updated_ratings_df = pd.concat([ratings, new_user_ratings], ignore_index=True)
    
    # Return the ID of the new user and the updated ratings dataframe
    return new_user_id, updated_ratings_df


def get_recommendation_svd(user_id, n=5):
    '''Give n recommendation to user_id'''
    
    all_books, user_books =  book_read(user_id)
    next_books = [book for book in all_books if book not in user_books]
    
    if n <= len(next_books):
        ratings = []
        for book in next_books:
            est = svd_model.predict(user_id, book).est
            ratings.append((book, est))
        ratings = sorted(ratings, key=lambda x: x[1], reverse=True)
        book_ids = [id for id, rate in ratings[:n]]
        return books[books.book_id.isin(book_ids)][['book_id', 'title', 'authors', 'year', 'pages', 'description', 'genres', 'average_rating', 'small_image_url']]
    else:
        print('Please reduce your recommendation request')
        return
    

def simple_recommender(books, n=6):
    v = books['ratings_count']
    m = books['ratings_count'].quantile(0.95)
    R = books['average_rating']
    C = books['average_rating'].median()
    score = (v/(v+m) * R) + (m/(m+v) * C)   
    books['score'] = score
    
    qualified  = books.sort_values('score', ascending=False)
    recommended_books = qualified[['book_id', 'title', 'authors', 'year', 'genres',
       'average_rating','small_image_url']].head(n)
    recommended_books_dict = recommended_books.to_dict(orient='index')
    return recommended_books_dict


user_ratings = {}
@app.route("/", methods=["POST", "GET"])
def home():
    error = False
    error_message = ""
    if request.method == "POST":
        try:
            title = request.form["movie_input"]
            rating = int(request.form["rating_input"])
            if title and rating:
                user_ratings[title] = rating
                flash(f"Successfully added [{title}]", "info")
            else:
                error = True
                error_message = "Title or rating is missing."
        except Exception as e:
            print(f"EXCEPTION AT HOME: {e}")
            error = True
            error_message = "Invalid rating input."
        print("User ratings:",user_ratings)
    get_recommendation(user_ratings)
    recommended_books = simple_recommender(books, 5)
    return render_template("home.html", error=error, error_msg=error_message, recommended_books=recommended_books)


@app.route("/recommend",methods=["POST"])
def get_recommendation(user_ratings):
    new_user_id, updated_ratings_df = get_new_user_id(user_ratings)
    print(new_user_id)
    recommended_books = get_recommendation_svd(123)
    print(recommended_books)

@app.route("/genres/<genre>", methods=["GET"])
def genres(genre):
    genres_based_books = books[books.genres.str.contains(genre, case=False)].head(9)
    genres_based_dict = genres_based_books.to_dict(orient='index')
    # print(genres_based_dict)
    # genres_based_dict = json.loads(genres_based_dict)
    return render_template("genres.html", genre_books=genres_based_dict, genre=genre)


if __name__ == '__main__':
    app.run(debug = True)