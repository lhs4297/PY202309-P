from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
#from movie_recommender import *
#from data_reader import *
import pandas as pd

# 데이터 전처리
def prepare_movie_features(movies_df):
    # 딕셔너리인 경우 문자열로 변환
    movies_df['genre'] = movies_df['genre'].apply(lambda x: str(x) if isinstance(x, dict) else x)
    movies_df['director'] = movies_df['director'].apply(lambda x: str(x) if isinstance(x, dict) else x)
    
    # 문자열로 변환된 'genre'와 'director'를 LabelEncoder로 변환
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

    # movie 데이터 프레임 생성
    movie_df = pd.DataFrame([movie])
    
    # movie 특성 준비
    movie_features = prepare_movie_features(movie_df)
    
    # 평점 예측
    predicted_rating = model.predict(movie_features)
    return predicted_rating
