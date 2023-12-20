from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from movie_recommender import *
#from data_reader import *

# 데이터 전처리
def prepare_movie_features(movies_df):
    movies_df['genre'] = LabelEncoder().fit_transform(movies_df['genre'])
    movies_df['director'] = LabelEncoder().fit_transform(movies_df['director'])
    return movies_df[['genre', 'director']]

def prepare_user_history(user_history_df, movies_df):
    movie_features = prepare_movie_features(movies_df)
    user_history_df = user_history_df.join(movie_features, on='movieId')
    return user_history_df

def predict_user_rating(user_history_df, movie):

    # 사용자 과거 평점 학습
    model = RandomForestRegressor()
    model.fit(user_history_df.drop('rating', axis=1), user_history_df['rating'])


    # 평점 예측
    predicted_rating = model.predict([movie])
    return predicted_rating

