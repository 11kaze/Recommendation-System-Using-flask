from flask import Flask,request,render_template
import pickle
import csv
import pandas as pd
import numpy as np
import time
#--------------------------------------------------------------
ratings = pd.read_csv('ratings.csv', encoding='latin-1')
movies = pd.read_csv('movies.csv')
train = ratings.copy()

# pivot ratings into movie features
df_movie_features = train.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

dummy_train = train.copy()

dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x>=1 else 1)

# The movies not rated by user is marked as 1 for prediction. 
dummy_train = dummy_train.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(1)


movie_features = train.pivot(
    index='userId',
    columns='movieId',
    values='rating'
)

# Normalising the rating of the movie for each user around 0 mean

mean = np.nanmean(movie_features, axis=1)
df_subtracted = (movie_features.T-mean).T

# Finding cosine similarity
from sklearn.metrics.pairwise import pairwise_distances

# User Similarity Matrix
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0


user_correlation[user_correlation<0]=0


user_predicted_ratings = np.dot(user_correlation, movie_features.fillna(0))


user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
def generate(userId):
    result=user_final_rating.iloc[userId].sort_values(ascending=False)[0:6]
    movie_id = result.index
    final_result = pd.DataFrame()
    # creating movie list
    movie_data = pd.read_csv('movies.csv')
    for x in movie_id:
        names = movie_data[movie_data['movieId'] == x ]
        final_result = final_result.append(names,ignore_index=True)
    final_result.drop('movieId',axis='columns', inplace=True)
    final_result.reset_index()
    return final_result
#--------------------------------------------------------------
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("login.html")

with open('id_and_password.csv', mode='r') as infile:
    reader = csv.reader(infile)
    database = {rows[0]:rows[1] for rows in reader}   
database.pop('id')

@app.route('/form_login',methods=['POST','GET'])
def login():
    name1=request.form['username']
    pwd=request.form['password']
    if name1 not in database:
	    return render_template('login.html',info='Invalid User')
    else:
        if database[name1]!=pwd:
            return render_template('login.html',info='Invalid Password')
        else:
            final_list = generate(int(name1))
            return render_template('home.html',name=name1 , data = final_list)

@app.route('/logout',methods=['POST','GET'])
def logout():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug = True)
