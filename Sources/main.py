import pandas as pd
from surprise import Dataset, Reader

def main():
    recommender = MovieRecommender()
    #recommender.load_data('./moive_dataset/path', './ratings/path')
    recommender.train()
    recommendations = recommender.recommend_for_user()
    print(recommendations)

def loadData(self, movies_path, ratings_path):
        # pandas를 이용해 csv 파일 읽어오기
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)

        # 데이터 분석 코드

        print(movies.head())
        print(ratings.head())

        # surprise 라이브러리를 사용하기 위해 데이터셋 로드
        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

def userFavoriteMovie():
     # 유저가 선호하는 영화 정보 구하기 ( 방법 생각 해봐야함 )

def matchMovie():
     # 읽어 온 데이터셋과 사용자의 영화 적합도 계산하는 코드
     
if __name__ == "__main__":
    main()