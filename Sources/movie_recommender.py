# from surprise import Dataset, Reader, KNNBasic
# from surprise.model_selection import cross_validate
from data_reader import *
import pandas as pd
import pylab as pl
from fuzzywuzzy import process
from sklearn.cluster import KMeans
from sklearn import preprocessing
from mlxtend.frequent_patterns import apriori, association_rules

from itertools import combinations




# class MovieRecommender:
#     def __init__(self):
#         self.data = None
#         self.trainset = None
#         self.algo = KNNBasic()

#     def train(self):
#         self.trainset = self.data.build_full_trainset()
#         self.algo.fit(self.trainset)

#     def load_data(self, movies_path, ratings_path):
#         # pandas를 이용해 csv 파일 읽어오기
#         movies = pd.read_csv(movies_path)
#         ratings = pd.read_csv(ratings_path)


def apriori_encoding(r):
    if r <= 0:
        return 0
    elif r >= 1:
        return 1

# def user_based_collaborative_filtering(_input_movies, _movies_df, _ratings_df):
#     try:
#         _recommendations = []

#         # Create a user-movie matrix where each row represents a user and the columns represent movies.
#         user_movie_matrix = _ratings_df.pivot(index='userId', columns='title', values='rating').fillna(0)

#         # Calculate the cosine similarity between each pair of users.
#         user_similarity = cosine_similarity(user_movie_matrix)
#         user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

#         # For each input movie, find users who have rated this movie.
#         for movie in _input_movies:
#             if movie in user_movie_matrix.columns:
#                 similar_users = user_similarity_df[user_movie_matrix[movie] > 0].index.tolist()

#                 # For each similar user, find the movies they have rated highly and recommend them.
#                 for user in similar_users:
#                     recommended_movies = user_movie_matrix.loc[user][user_movie_matrix.loc[user] > 4].index.tolist()
#                     for recommended_movie in recommended_movies:
#                         if recommended_movie not in _input_movies and recommended_movie not in _recommendations:
#                             _recommendations.append(recommended_movie)

#     except Exception as e:
#         print(f"Error in user_based_collaborative_filtering: {e}")

#     return _recommendations

def do_apriori(_input_movies, _movies_df, _ratings_df):
    try:
        # Internal variables
        _apriori_result = []

        """ Remove the Nan title & join the dataset """
        Nan_title = _movies_df['title'].isna()
        _movies_df = _movies_df.loc[Nan_title == False]

        _movies_df = _movies_df.astype({'id' : 'int64'})
        df = pd.merge(_ratings_df, _movies_df[['id', 'title']], left_on='movieId', right_on='id')
        df.drop(['timestamp', 'id'], axis=1, inplace=True)

        # Check if df is empty
        print(f"df shape: {df.shape}")
        if df.empty:
            print("df is empty")
            return _apriori_result

        """ Prepare Apriori
            row : userId | col : movies """
        df = df.drop_duplicates(['userId', 'title'])
        df_pivot = df.pivot(index='userId', columns='title', values='rating').fillna(0)
        df_pivot = df_pivot.astype('int64')
        df_pivot = df_pivot.applymap(apriori_encoding).astype(bool)
        # print(df_pivot.head())

        # Check if df_pivot is empty
        print(f"df_pivot shape: {df_pivot.shape}")
        if df_pivot.empty:
            print("df_pivot is empty")
            return _apriori_result

        """ A-priori Algorithm """
        #calculate support and eradicate under min_support
        frequent_items = apriori(df_pivot, min_support=0.07, use_colnames=True)
        print(frequent_items.head())

        # Check if frequent_items is empty
        print(f"frequent_items shape: {frequent_items.shape}")
        if frequent_items.empty:
            print("frequent_items is empty")
            return _apriori_result

        # using association rules, compute the other parameter ex) confidence, lift ..
        association_indicator = association_rules(frequent_items, metric="lift", min_threshold=1)

        # Check if association_indicator is empty
        print(f"association_indicator shape: {association_indicator.shape}")
        if association_indicator.empty:
            print("association_indicator is empty")
            return _apriori_result

        # sort by order of lift
        df_lift = association_indicator.sort_values(by=['lift'], ascending=False)
        # print(df_res.head())

        """ Start recommendation """
        for r in range(len(_input_movies), 0, -1):
            for selected_movies in combinations(_input_movies, r):
                df_selected = df_lift[df_lift['antecedents'].apply(lambda x: set(x) == set(selected_movies))]
                df_selected = df_selected[df_selected['lift'] > 1.0]
                df_selected.sort_values(by='lift', ascending=False, inplace=True)  # Sort by lift in descending order

                recommended_movies = df_selected['consequents'].values

                for movie in recommended_movies:
                    for title in movie:
                        if title not in _input_movies and title not in _apriori_result:
                            _apriori_result.append(title)
                            if len(_apriori_result) == 5:  # Stop when 5 movies are recommended
                                break

                if len(_apriori_result) == 5:
                    break
            if len(_apriori_result) == 5:
                break

    except Exception as e:
        print(f"Error in do_apriori: {e}")

    return _apriori_result



def do_kmeans(_apriori_result, _input_movies, _movies_df):
    try:
        # record all clusters in _input_movies
        clusters = []
        _kmeans_result = []

        numeric_df = _movies_df[['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count', 'title']]

        numeric_df.isnull().sum()
        numeric_df.dropna(inplace=True)
        # print(df_numeric['vote_count'].describe())

        """cut off the movies' votes less than 25"""
        df_numeric = numeric_df[numeric_df['vote_count'] > 25]

        # Normalize data - by MinMax scaling provided by sklearn
        minmax_processed = preprocessing.MinMaxScaler().fit_transform(df_numeric.drop('title', axis=1))
        df_numeric_scaled = pd.DataFrame(minmax_processed, index=df_numeric.index, columns=df_numeric.columns[:-1])

        """Apply K-means clustering"""
        # make elbow curve to determine value 'k'
        num_cluster = range(1, 20)
        kmeans = [KMeans(n_clusters=i) for i in num_cluster]
        score = [kmeans[i].fit(df_numeric_scaled).score(df_numeric_scaled) for i in range(len(kmeans))]

        # print elbow curve
        pl.plot(num_cluster, score)
        pl.xlabel("Number of clusters")
        pl.ylabel("Score")
        pl.title("Elbow curve")
        #plt.show()  # maybe k=4 is appropriate

        # Fit K-means clustering for k=5
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(df_numeric_scaled)  # result is kmeans_label

        # write back labels to the original numeric data frame
        df_numeric['cluster'] = kmeans.labels_
        # print(df_numeric.head())

        # Search all clusters in user selected movies
        for movie1 in _input_movies:
            try:
                cluster_candid = df_numeric.loc[df_numeric["title"] == movie1, 'cluster'].values[0]
                # print(cluster_candid)
                clusters.append(cluster_candid)
            except IndexError as e:
                msg = "There is No cluster in movie [" + movie1 + ']'
                ErrorLog(msg)
                #print(msg)

        # Filtering movies that are not in clusters
        for movie2 in _apriori_result:
            try:
                cluster_tmp = df_numeric.loc[df_numeric["title"] == movie2, 'cluster'].values[0]
                if cluster_tmp in clusters:
                    _kmeans_result.append(movie2)
            except IndexError as e:
                msg = "There is No cluster in movie [" + movie2 + ']'
                ErrorLog(msg)
                #print(msg)
    except Exception as e:
        print(f"Error in do_kmeans: {e}")

    return _kmeans_result



