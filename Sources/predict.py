from sklearn.ensemble import RandomForestRegressor

def predict_user_rating(user_history_df, movie):

    # 사용자 과거 평점 학습
    model = RandomForestRegressor()
    model.fit(user_history_df.drop('rating', axis=1), user_history_df['rating'])


    # 평점 예측
    predicted_rating = model.predict([movie])
    return predicted_rating
