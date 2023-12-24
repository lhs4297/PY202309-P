from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def prepare_movie_features(movies_df):
    # 딕셔너리인 경우 문자열로 변환
    movies_df['title'] = movies_df['title'].apply(lambda x: str(x) if isinstance(x, dict) else x)
    movies_df['vote_average'] = movies_df['vote_average'].apply(lambda x: str(x) if isinstance(x, dict) else x)
    
    # 문자열로 변환된 'title'와 'vote_average'를 LabelEncoder로 변환
    movies_df['title'] = LabelEncoder().fit_transform(movies_df['title'])
    movies_df['vote_average'] = LabelEncoder().fit_transform(movies_df['vote_average'])
    
    return movies_df[['title', 'vote_average']]

def prepare_user_history(user_history_df, movies_df):
    movie_features = prepare_movie_features(movies_df)
    # 'id'를 기준으로 user_history_df와 movie_features를 합침
    user_history_df = pd.merge(user_history_df, movies_df, left_on='movieId', right_on='movieId')
    return user_history_df

def predict_user_rating(user_history_df, movies_df):
    for column in movies_df.columns:
        print(f"{column}: {movies_df[column].iloc[0]}")

    model = RandomForestRegressor()

    user_history_df = prepare_user_history(user_history_df, movies_df)

    # 사용자 과거 평점 학습
    model.fit(user_history_df.drop('rating', axis=1), user_history_df['rating'])

    # movie 데이터 프레임 생성
    # 'title' 컬럼에 대한 처리
    movies_df['title'] = movies_df['title'].apply(lambda x: str(x) if isinstance(x, list) else x)

    # movie 특성 준비
    movie_features = prepare_movie_features(movies_df)

    # 평점 예측
    predicted_rating = model.predict(movie_features)
    return predicted_rating
