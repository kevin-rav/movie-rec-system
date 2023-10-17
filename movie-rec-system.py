import pandas as pd 
import numpy as np
import re 
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def menu():
    print("")
    print("---------------------------")   
    print("Movie Recommendation System")
    print("---------------------------")
    print("")
    print("   1. Search Movies By Title")
    print("   2. Movie Recommendations")
    print("   3. Exit")
    print("")
    option = input("Enter option (1-3): ")
    return option

def separate_year(title):
    return title[title.find("(")+1:title.find(")")]

def clean_title(title):
    if ", The (" in title:
        split = title.split(',')
        title = "The " + split[0]
    title = re.sub(" \(.*?\)", "", title)
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices][::-1]
    return results

def similar_movies(movie_id): 
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .1]

    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

movies = pd.read_csv("movies.csv")
movies["clean_title"] = movies["title"].apply(clean_title)
movies["year"] = movies["title"].apply(separate_year)

vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

ratings = pd.read_csv("ratings.csv")

option = menu()
clear = lambda: os.system("clear")

while option != "3":

    if option == "1":
        clear()
        movie_name = input("Enter Movie Name: ")
        movie_titles = search(movie_name)["clean_title"].tolist()
        movie_years = search(movie_name)["year"].tolist()
        print("")
        print("Movies with Similar Title: ")
        for i in range(5):
            print(str(i+1) + ". " + movie_titles[i] + " (" + movie_years[i] + ")")
    
    elif option == "2":
        clear()
        movie_name = input("Enter Movie Name: ")
        movie_ids = search(movie_name)["movieId"].tolist()
        recommended_movies = similar_movies(movie_ids[0])["title"].apply(clean_title).tolist()
        print("")
        print("You should watch: ")
        i = 1
        if len(recommended_movies) > 0:
            for movie in recommended_movies:
                print(str(i) + ". " + movie)
                i += 1
            i = 0
        else:
            print("Movie not found")

    else:
        clear()
        print("Input invalid. Try again")

    option = menu()

print("Goodbye")


