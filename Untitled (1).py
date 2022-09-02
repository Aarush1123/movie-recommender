#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


credits=pd.read_csv("tmdb_5000_credits.csv")
movies=pd.read_csv("tmdb_5000_movies.csv")
movies.head(10)


# In[5]:


credits.head(10)


# In[6]:


movies = movies.merge(credits,on='title')


# In[7]:


movies.head(1)


# In[8]:


movies=movies[['movie_id', 'title' ,'overview', 'genres' , 'keywords' , 'cast', 'crew']]


# In[9]:


movies.head(1)


# In[10]:


movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.iloc[0].genres


# In[14]:


import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[15]:


movies['genres']=movies['genres'].apply(convert)


# In[16]:


movies.head(1)


# 

# 

# In[17]:


movies['keywords']=movies['keywords'].apply(convert)


# In[18]:


movies.head(1)


# 

# 

# In[19]:


import ast
def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[20]:


movies['cast']= movies['cast'].apply(convert3)


# In[21]:


movies.head(1)


# In[22]:


import ast
def fetch_director(obj):
    L=[]
    
    for i in ast.literal_eval(obj):
        if i['job'] =='Director':
            L.append(i['name'])
            break
    return L


# In[23]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[24]:


movies.head(1)


# In[25]:


movies['overview']=movies['overview'].apply(lambda x: x.split())


# In[26]:


movies.head(2)


# In[27]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[28]:


movies.head(5)


# In[29]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['cast']


# In[30]:


movies.head(1)


# In[31]:


new_df=movies[['movie_id', 'title','tags']]


# In[32]:


new_df


# In[33]:


new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))


# In[34]:


new_df


# In[35]:


new_df['tags']=new_df['tags'].apply(lambda x: x.lower())


# In[36]:


new_df.head(1)


# In[37]:


import nltk


# In[38]:


from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()


# In[39]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[40]:


new_df['tags']=new_df['tags'].apply(stem)


# In[41]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[42]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[43]:


vectors


# In[50]:


from sklearn.metrics.pairwise import cosine_similarity


# In[51]:


similarity=cosine_similarity(vectors)


# In[52]:


sorted (list(enumerate(similarity[0])),reverse=True , key= lambda x:x[1])[1:6]


# In[57]:


def recommend (movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movie_list=sorted (list(enumerate(distances)),reverse=True , key= lambda x:x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        


# In[58]:


recommend('Avatar')


# In[ ]:




