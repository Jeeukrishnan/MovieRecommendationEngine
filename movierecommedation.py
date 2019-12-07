import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#helper function to use when needed
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]


#1read data
 
df=pd.read_csv("moviedata.csv")
print (df.columns)

#2slection of features

features=['keywords','cast','genres','director']

#3create a coloumn in dataframe which combines all features in column
for feature in features:
    df[feature]=df[feature].fillna(' ') #fillna function will replace all NaN with empty string

def combine_features(row):
    try :
     return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
    except :
        print ("Error :" ,row)
df["combine_features"]=df.apply(combine_features,axis=1)  #commbine all rows to make a col.

print (df["combine_features"])

#4 count matrix for this new coloumn
cv = CountVectorizer()
cv_matrix=cv.fit_transform(df["combine_features"])


#5compute cosine similarity
cosine_sim=cosine_similarity(cv_matrix)

print (cosine_sim)

movie_user_likes="Avatar"  ##Given

#6Get index of this movie from its tittle
movie_index=get_index_from_title(movie_user_likes)
  
similar_movies=list(enumerate(cosine_sim[movie_index])) 
 #it will give the index values which are similar to given index movie ,it will return the row of avatar movie
 
#7get the sorted list  in descending order
sorted_similar_movies=sorted(similar_movies,key= lambda x:x[1],reverse=True)
#x:x[1] is used so that it get sorted according to second element of the tuple
#first element is the index in the list and second element is similarity in the tuple of sorted_similar_movies list


#print tittle of first 50 movies
i=0
for movie in sorted_similar_movies:
     print (get_title_from_index(movie[0]))
     i=i+1
     if i>50 :
         break


