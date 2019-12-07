from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

text=["London Paris London","Paris London Paris"]
#CountVectorizer to count the frequency of the words
cv = CountVectorizer()
cv_matrix=cv.fit_transform(text)
print(cv_matrix.toarray())

#count similarity
similarity_scores=cosine_similarity(cv_matrix)
print(similarity_scores)