import pandas as pd
import warnings; warnings.simplefilter('ignore')
from movie_recommender import *
from data_reader import *



def main(input_movies):
    final_result = ""
    final_result += "Selected movies (5 movies) : " + ",".join(input_movies) + "\n\n"
    print(final_result)

    # csv파일을 읽어온다
    movies_df = pd.read_csv('./movie_dataset/movies_metadata.csv')
    ratings_df = pd.read_csv('./movie_dataset/ratings_small.csv')

    # Drop the trash(error) data
    movies_df = drop_trash_data(movies_df)

    # a-priori & k-means를 이용해서 추천을 해준다
    apriori_result = do_apriori(input_movies, movies_df, ratings_df)
    kmeans_result = do_kmeans(apriori_result, input_movies, movies_df)


    # 추가해야될 부분
    # kmeans_result를 문자열로 변환
    # 영화 목록을 쉼표로 구분
    # 저장
    
    # 결과를 result.txt파일로 출력해줌
    print(final_result)
    f = open("result.txt", "w")
    f.write(final_result)
    f.close()

    return final_result




if __name__ == '__main__':
    #미구현 대체 : input_movies를 임시로 지정
    input_movies = ['Toy Story', 'Jumanji', 'Grumpier Old Men', 'Waiting to Exhale', 'Father of the Bride Part II']
    main(input_movies)