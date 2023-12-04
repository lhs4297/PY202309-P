import time
from ast import literal_eval



def ErrorLog(error):
    current_time = time.strftime("%Y.%m.%d/%H:%M:%S", time.localtime(time.time()))
    with open("Log.txt", "a") as f:
        f.write(f"[{current_time}] - {error}")

def find_movieId(_input_movies, _movies_df):
    indices =[]
    for movie in _input_movies:
        index = _movies_df.loc[_movies_df["original_title"] == movie, 'id'].values[0]
        indices.append(index)
    return indices

def drop_trash_data(_movie_df):
    _movie_df.drop(_movie_df.index[19730], inplace=True)
    _movie_df.drop(_movie_df.index[29502], inplace=True)
    _movie_df.drop(_movie_df.index[35585], inplace=True)
    return _movie_df
