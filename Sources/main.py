import pandas as pd
import warnings; warnings.simplefilter('ignore')
from movie_recommender import *
from data_reader import *
import numpy as np




def main(input_movies):
    print("마음에 들었던 영화를 입력하세요: ")
    #input_movies = input()
    final_result = ""
    final_result += "Selected movies (5 movies) : " + ",".join(input_movies) + "\n"
    print(final_result)

    # csv파일을 읽어온다
    movies_df = pd.read_csv('C:/Users/user/Desktop/현승/23_1학기_학교/C 및 PY/PY_프로젝트/PY202309-P/Sources/movie_dataset/movies_metadata.csv')
    ratings_df = pd.read_csv('C:/Users/user/Desktop/현승/23_1학기_학교/C 및 PY/PY_프로젝트/PY202309-P/Sources/movie_dataset/ratings_small.csv')

    # Drop the trash(error) data
    movies_df = drop_trash_data(movies_df)

    # a-priori & k-means를 이용해서 추천을 해준다
    apriori_result = do_apriori(input_movies, movies_df, ratings_df)
    kmeans_result = do_kmeans(apriori_result, input_movies, movies_df)

    # Add results to final_result
    final_result += "\nApriori recommendations:\n" + ", ".join(apriori_result) + "\n"
    final_result += "\nK-means recommendations:\n" + ", ".join(kmeans_result) + "\n"

    # 예상 평점 계산
    user_history_df = ratings_df[ratings_df['userId'] == 1]  # 사용자 ID를 지정합니다.
    for movie in kmeans_result:
        movie_info = movies_df[movies_df['title'] == movie]
        predicted_rating = predict_user_rating(user_history_df, movie_info, movies_df)
        final_result += f"\nThe predicted rating for {movie} is {predicted_rating[0]}\n"

    # 결과를 result.txt파일로 출력해줌
    print(final_result)
    f = open("result.txt", "w")
    f.write(final_result)
    f.close()

    return final_result





if __name__ == '__main__':
    #input_movies를 임시로 지정
    input_movies = ['Toy Story', 'Jumanji', 'Grumpier Old Men', 'Transformers', 'Batman Forever']
    main(input_movies)