```python
import pandas as pd
import numpy as np

from typing import List, Dict
```


```python
from sklearn.feature_extraction.text import TfidfVectorizer
```


```python
pwd()
```




    '/Users/root947/Desktop/Recommender_system'




```python
cd Recommender_system
```

    /Users/root947/Desktop/Recommender_system



```python
#Read data into dataframe
songs = pd.read_csv('songdata.csv')
# Inspect the data
songs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>song</th>
      <th>link</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>ABBA</td>
      <td>Ahe's My Kind Of Girl</td>
      <td>/a/abba/ahes+my+kind+of+girl_20598417.html</td>
      <td>Look at her face, it's a wonderful face  \nAnd...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>ABBA</td>
      <td>Andante, Andante</td>
      <td>/a/abba/andante+andante_20002708.html</td>
      <td>Take it easy with me, please  \nTouch me gentl...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>ABBA</td>
      <td>As Good As New</td>
      <td>/a/abba/as+good+as+new_20003033.html</td>
      <td>I'll never know why I had to go  \nWhy I had t...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>ABBA</td>
      <td>Bang</td>
      <td>/a/abba/bang_20598415.html</td>
      <td>Making somebody happy is a question of give an...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>ABBA</td>
      <td>Bang-A-Boomerang</td>
      <td>/a/abba/bang+a+boomerang_20002668.html</td>
      <td>Making somebody happy is a question of give an...</td>
    </tr>
  </tbody>
</table>
</div>




```python
songs.describe(include = 'all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>song</th>
      <th>link</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>57650</td>
      <td>57650</td>
      <td>57650</td>
      <td>57650</td>
    </tr>
    <tr>
      <td>unique</td>
      <td>643</td>
      <td>44824</td>
      <td>57650</td>
      <td>57494</td>
    </tr>
    <tr>
      <td>top</td>
      <td>Donna Summer</td>
      <td>Have Yourself A Merry Little Christmas</td>
      <td>/k/kenny+rogers/if+i+could+hold+on+to+love_201...</td>
      <td>Chestnuts roasting on an open fire  \nJack Fro...</td>
    </tr>
    <tr>
      <td>freq</td>
      <td>191</td>
      <td>35</td>
      <td>1</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# resample only 5000 songs out of 57650 songs available
songs = songs.sample(n = 5000)
```


```python
songs.describe(include = 'all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>song</th>
      <th>link</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>unique</td>
      <td>603</td>
      <td>4763</td>
      <td>5000</td>
      <td>4998</td>
    </tr>
    <tr>
      <td>top</td>
      <td>Christmas Songs</td>
      <td>A Song For You</td>
      <td>/c/conway+twitty/halfway+to+heaven_20213954.html</td>
      <td>Baby here I stand before you  \nWith my heart ...</td>
    </tr>
    <tr>
      <td>freq</td>
      <td>23</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Replace \n present in the text
songs['text'] = songs['text'].str.replace('\n', '' )
```


```python
# Initialize tfidf vectorizer
tfidf = TfidfVectorizer(analyzer = 'word', stop_words= 'english')

# this is used for calculating tf (term frequency) and idf (inverse document frequency)

#tf = number of occurence of that word/ total number of words
#idf = log(total number of documents/no of documents containg the term)

# now we are going to calculate the TF-IDF score for each song lyric word by word i.e. TF * IDF
#tfidf.get_feature_names()
```


```python
# Fit and transform 
tfidf_matrix = tfidf.fit_transform(songs['text'])
```


```python
tfidf_matrix.shape 
# from the shape, we see that the matrix has 
# as rows all the 5000 songs and each words 
# importance corresponding to the song is given
```




    (5000, 24591)




```python
tfidf_matrix.todense()
```




    matrix([[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]])




```python
# calculate similarity of one lyric to the next (using euclidean distance or cosine similarity)
# we use cosine because we are not only interested in the magnitude of the term frequency but also the angle between the lyrics; small angle means more similar
# (refer to http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/)
from sklearn.metrics.pairwise import cosine_similarity

#cosine_similarity(tfidf_matrix[0:1], tfidf_matrix) # to see the similarity of the first song with all the other songs; in the output, first value is 1 because the song is being compared to itself 
cosine_similarities = cosine_similarity(tfidf_matrix)
```


```python
cosine_similarities[0].argsort()[:-50:-1] # argort gives you the position (starting from 0) rather than the value itself
# getting the column numbers which essentialy represent the song number of the 50 most similar songs to the song[0]

```




    array([   0, 1803, 3205, 3791, 1757, 2384, 1442, 1535, 2579,  865, 4469,
           4608,  497, 2815, 1516, 1204,  132, 2695, 3466,  835, 1521, 1957,
           4254, 4906,   45,  656, 4123, 1871, 4910,   22, 1491,  775, 2222,
            343, 2044, 3973,  831, 1749, 3953, 1245, 4672, 4162, 1454, 2574,
             97,  213, 1449, 2816, 1507])




```python
# creating a dictionary to store for each song 50 most similar song
my_dict = {}

for i in range(len(cosine_similarities)):
    similar_indices = cosine_similarities[i].argsort()[:-50:-1] # returns the indexes of top 50 songs
    
    # setting the key as the song name
    song_name = songs['song'].iloc[i] 
    # setting the value as three items i.e (1) similarity score, (2) song name, (3) artist name
    my_dict[song_name] = [(cosine_similarities[i][x], songs['song'].iloc[x], songs['artist'].iloc[x]) for x in similar_indices][1:]
```


```python
# Testing whether this dictionary works ro give most similar songs for Song number 10
getforsong = songs['song'].iloc[10]
getforsong
my_dict.get(getforsong)[:3]
```




    [(0.5066125791275065, 'Love Me Now', 'John Legend'),
     (0.5060429641432027, 'For The Girl Who Has Everything', "'n Sync"),
     (0.47467050528723503, "Lookin' For That Girl", 'Tim McGraw')]




```python
# Let us use this dictionary to present the recommendations

def get_recommendation(ques):
            # Get song to find recommendations for
            song = ques['song']
            # Get number of songs to recommend
            number_songs = ques['no_of_songs']
            # Get the number of songs most similars from matrix similarities
            recom_song = my_dict.get(song)[:number_songs]
            # print each item
            print(f"The recommended songs for {song} by {songs.loc[songs['song'] == song, 'artist'].iloc[0]} are")
            for i in range(number_songs):
                print(f"{recom_song[i][1]} by {recom_song[i][2]} with similarity score {recom_song[i][0]} ")
#             print(recom_song)
```


```python
# Create a dictionary to pass as input

ques = {
    "song" : songs['song'].iloc[1104],
    "no_of_songs" : 4
}

get_recommendation(ques)
```

    The recommended songs for Working Class Hero by Marilyn Manson are
    Hero by Mariah Carey with similarity score 0.4161157147759222 
    Barrier Reef by Old 97's with similarity score 0.31360076636387807 
    Heroes by Helloween with similarity score 0.21100304905514192 
    Clampdown by Indigo Girls with similarity score 0.20893836777075372 



```python
# to find the artist name given the song name
# songs.loc[songs['song'] == "Pieces Of A Dream", 'artist'].iloc[0]
```
