# from surprise import Dataset, Reader, KNNBasic
# from surprise.model_selection import cross_validate
from data_reader import *
import pandas as pd
import pylab as pl
from fuzzywuzzy import process
from sklearn.cluster import KMeans
from sklearn import preprocessing
from mlxtend.frequent_patterns import apriori, association_rules




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
        # print(frequent_items.head())

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
        for selected_movie in _input_movies:
            num = 0
            df_selected = df_lift[df_lift['antecedents'].apply(lambda x: len(x) == 1 and next(iter(x)) == selected_movie)]
            df_selected = df_selected[df_selected['lift'] > 1.0]
            recommended_movies = df_selected['consequents'].values

            for movie in recommended_movies:
                for title in movie:
                    if title not in _apriori_result and num < 10:
                        _apriori_result.append(title)
                        num += 1

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



