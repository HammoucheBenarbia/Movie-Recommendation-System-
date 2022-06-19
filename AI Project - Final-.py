#!/usr/bin/env python
# coding: utf-8

# # Student :                Hammouche Ben arbia 

# # Contents 

# # Overview of  Recommendation systems 

# + Collaborative Filtering

# + Content-Based systems

# + Hybrid recommenders

# # Building Movies recommendation system

# + N Top Recommender 

# + Cosine similarity

# + Matrix Factorization 

# + Other recommendations techniques with Surprise Package 

# # 1- Overview of  Recommendation systems 

# Recommendation systems are ensemble techniques of suggesting products or services to potential buyers. Recommendation systems can be categorized in many categories:
# 
# # A) Collaborative Filtering:
# 
# This technique uses users’ past activity data to look at the resemblance between users (user based collaborative filtering) or items (item based collaborative filtering). The idea behind this technique is that if users have similar preferences, then we can recommend products that one user liked for another similar user who didn’t try the product yet.  For item-based recommendation if two items have been rated similarly by a group of people then the two items must be similar. Consequently, if a customer liked one item, then there is high chance that he will like the other item too. 
# 
# # b) Content-Based systems:
# This technique differs from collaborative filtering in its not reliance in users’ past activity data, this technique uses user profile and its data that it has on items. In other words, its algorithms will ask to user to rate randomly items and based on users’ rating it will find similar items to recommend. 
# 
# # c) Hybrid recommenders: 
# This technique used a combination of different recommendation system including the ones that we stated previously. This technique combines techniques in way that advantage from one algorithm will cover a disadvantage from another algorithm and come up with a robust model. 
# 
# 
# # 2- Building Movies recommendation system
# In this project we will try to build a movie recommendation system using movie lens 100k data set. because of the limits of this project we will use only some thechniques of Collaborative filletering as following: 
# 

# - Top N weighted rating recommender 

# - Cosine similarity 

# - Matrix Factorization 

# - Other recommendations techniques with Surprise Package ( SVD,KNN,NMF)

# In[3]:


# Import packages and libraries like (pandas and numpy)
import os
os.getcwd()
os.chdir("Desktop")
import pandas as pd 
import numpy as np


# # Load the data 

# In[4]:


#read the data sets and check the first few rows 
ratings=pd.read_csv("ratings.csv")
movies=pd.read_csv("movies.csv")
print(ratings.head(10))
print(movies.head(10))


# In[5]:


#Drop the timestamp from the data set 
ratings=ratings.drop(["timestamp"],axis=1)
print(ratings.head(10))


# In[6]:


#Merge ratings and movies datasets
ratings_movies=pd.merge(ratings,movies,on="movieId")
print(ratings_movies.head())


# In[35]:


#Check for missing values in the dataset 
print(ratings_movies.isnull().sum())


# In[37]:


#Create a dictionary to map movie id with movies titles 
movie_dic={}
for i in (range(100836)):
    movieID=ratings_movies.loc[i,"movieId"]
    movie_dic[movieID]= ratings_movies.loc[i,"title"] 


# # N Top Recommender 
# This recommender is the simplest comparing to others, it recommends movies based on specific metric. the reason why we use a specific metric instead of the rating already available on the data set is because if we take the rating, we will get wrong ranking for example a movie that was rated 5 stars by 10 people will be ranked before a movie that was giving 4.5 by 10000 people. to deal with this issue we will choose a metric that takes on consideration both the rating and the number of ratings to determine the best ranking. In this project Weighted rating will be used.
# 
# # Weighted Rating 
# 
# This metric is used by IMDB platform to rank the movies, The Weighted rating can be computed with this following equation:
# 
#   

# # Weighted Rating (𝑊𝑅)=(V/(V+M))𝑅+(M/(V+M))𝐶
# where:

# 𝑅 = average rating for the movie (mean)

# M = minimum votes required

#  V = number of ratings for the movie (votes)

#  𝐶 = the mean vote across the whole rating column

# In[14]:


#Compute the mean rating for each movie ID 
R=ratings_movies.groupby(["movieId"])["rating"].mean().reset_index()
print(R)


# In[145]:


# Compute the number of ratings for each movie ID 
V=ratings_movies.groupby(["movieId"])["rating"].count().reset_index()
print(V)


# In[16]:


# The mean of  all movie ratings 
C=ratings_movies.rating.mean()
print(C)


# The Weighted rating  can be computed with this following equation:
# 
# Weighted Rating (𝑊𝑅)=(V/(V+M))𝑅+(M/(V+M))𝐶
# where:
# 𝑅 = average rating for the movie (mean)
# V = number of ratings for the movie (votes)
# M = minimum votes required
# 𝐶 = the mean vote across the whole rating column 

# In[17]:


# Create a function " ranking" to compute the Weighted ratings 
def ranking (v,r):
    WR=((v/v+1)*r)+((1/v+1)*C)
    return WR
# The value of m (minimum number of votes required for the movie to be in the cahrt is 1 vote)


# The following code will create two lists, the first one for number of ratings and the second for the average rating for each movie. theses lists will be used later for computing the weighting rate for each movie 

# In[38]:


# Create two lists number of votes (Vlist) and the mean rating of the movie
vlist=[]
rlist=[]
for k in (range(9724)):
    v_=V.iloc[k,1]
    r_=R.iloc[k,1]
    vlist.append(v_)
    rlist.append(r_)


# The following code will be used to compute the weighted rating for each movie. 

# In[39]:


#Compute the weighted rate for each movie 
List=[] # Empty list to append WR 
movieId_={} # Create dictionary to map between each score and its index
count=0 # this will be used to creat fixed indexing 
for j,k in zip(vlist,rlist):
    x=ranking(j,k)
    List.append(x)
    movieId_[x]=count
    count+=1
List.sort(reverse=True)


# The following code generate Top 50 recommender by looping through list of scores that we created previously then get the index of the movie and finaly using movie_dict to track back the movie title 

# In[302]:


#generate N Top recommender 
recommendation_list=[]
for y in (List):# Loop through sorted list of scores 
    indx=1
    indx=movieId_[y]
    M_indx=V.iloc[indx,0]
    M=movie_dic[M_indx]
    if M not in recommendation_list:
        recommendation_list.append(M)
    else:
        pass
print("The TOP movies TO recommend based on WR are:","\n")
print(*recommendation_list[:50],sep="\n")


# # Cosine similarity 

# As stated previously Collaborative filltering uses users' historical activities to find similar users and assume that simmilar users will give same ratings to items . One of a known metrics to gauge similiratity is Cosine similiraty

# - Cosine similarity gauges the difference between two vectors. in the movie rating example thses each vector represents user's preferences. Therefore similar vectors have now space between them the angle is close to zero and by consequanve the cosine of zero is 1 then the more cosine similarity has higher value the most the users are similar  

# - Cosine similarity is defined by this formula :

# ![Screen%20Shot%202021-12-10%20at%206.19.51%20PM.png](attachment:Screen%20Shot%202021-12-10%20at%206.19.51%20PM.png)

# ![Screen%20Shot%202021-12-10%20at%206.30.09%20PM.png](attachment:Screen%20Shot%202021-12-10%20at%206.30.09%20PM.png)

# In[298]:


# Create Cosine simmilarity function
def cosin_sim (U1,U2): # define two users U1 and U2 
    User=[] # List to append The product of users' magnitude 
    u1_=[] # List to append first user vector (rating)
    u2_=[] # List to append second user vector (rating)
    # put a condition for users length the be the same 
    if len(U1)>len(U2): 
        U1=U1[:len(U2)]
    elif len(U1)< len(U2):
        U2=U2[:len(U1)]
    for i,j in zip(U1,U2):# loop throught users' ratings  
        u1_.append(i**2) # square the vectors 
        u2_.append(j**2)
        User.append(i*j) # compute dot product of two vectors    
    U1_U2=sum(User) # The dot product of two vectors 
    U1_=sum(u1_)**0.5 # vector's magnitude 
    U2_=sum(u2_)**0.5

    return U1_U2/(U1_*U2_) # cosine 


# In[299]:


#create function to recommend users based on cosin simmilarity 
def recomm_ (user_id,dataset,number): # define inputs where number is how many movies we will recommende 
    score_user={} # dict to map cosine value with user id 
    userid_2=set(ratings_movies.userId.values) # create set that have unique user Ids 
    similarity=[] # cosine similarity list 
    user_1=dataset.loc[dataset["userId"]==user_id] # select  user for which we will recommend 
    rat_1=user_1.rating.values # Get rating of user 1 
    for i in userid_2:
        rat2=[] 
        if i==user_id:
            pass
        else:
            user=dataset.loc[dataset["userId"]==i] # get ratings for a specific user 
            rat_2=user.rating.values
            score=cosin_sim(rat_1,rat_2) # compute similarity between user_1 (input in the fuction ) and other users 
            score_user[score]=i # map cosine value with user 
            similarity.append(score) # append all cosine values in a list 
    similarity.sort(reverse=True) # sort cosine list list from high to low 
    movies_to_recommend=[] # list of movies to recommend 
    movie_list_user_1=user_1.title.values # list of movies watched by user_1 
    for j in similarity[:number]: # get number of similar users desired 
        indx=score_user[j] # get similar user index 
        # we will recommend movies that similar users gave a rating of   more than 3 
        movies_liked=dataset.loc[(dataset["userId"]==indx)&(dataset["rating"]>3)] 
        movie_title_liked=movies_liked.title.values
        # check movies liked by similar users but not have been tried by user_1 
        for movie_L in movie_title_liked: 
            if movie_L not in movie_list_user_1: 
                movies_to_recommend.append(movie_L)
            else:
                pass                       
    
    print("List of movies to recommend:\n")
    print( *movies_to_recommend[:10],sep="\n" )
    return  


# In[303]:


# list of 10 recommended movies for userID=34
recomm_(34,ratings_movies,10)


# In[305]:


# list of 10 recommended movies for userID=34
recomm_(1,ratings_movies,3)


# # Matrix Factorization 

# One of the effecient method of implementing collaborative filletring is matrix factorization. Unlike other previous method Matrix maktorization is used to predict a rating that user will give to an item. The idea behind Matrix factorization can be breaked down as following: 

# - Users didn't rate all the Items,therefore it will be items that haven't been rated by some users 

# - These missing ratings can be filled with zeros 

# + There are hidden patterns between items and users these patterns are called latent feutures, by using  matrix factorization technique we can find a numeric measurements for these features and use them to predict ratings 

# # Mathematic formulation of matrix factorization 

# The mathematic concept behind matrix factorization is using the defined matrix (M) which contains a set of users and items with size USERS X ITEMS to find latent features K, Matrix P , and matrix Q, where Matrix P with size Users X K represents the association between users and latent features (K) and Matrix Q represents  the association between Items and K . after performing dot product of P and Q we will get prediction of M.  

# ![Screen%20Shot%202021-12-09%20at%205.06.52%20PM.png](attachment:Screen%20Shot%202021-12-09%20at%205.06.52%20PM.png)

# To get the two matrices Q and P we initialize the values randomly then we compute the predicted rating.  from these predictions we calculate the errors between actuals and predicted and use gradient descent method to update the Q and P by minimizing the errors until we get better predictions with minimum errors.
# 

# # Biases 

# Predicting rating by using only latent features it's not enough, to reflect the real world ratings we need to add biases there three biases :

# - User Biase : users tend to rate movies higher or lower than others 

# - Item Biase : some items tend to recieve higher or lower  ratings comparing to other similar movies 

# - Global Biase : which can be representes by the average rating 

# BY adding biases the final prediction formula is :

# #Prediction = Global Biase + User Biase + Item Biase + P.dot Q 

# # Implementing Matrix factorization 

# # First we split the data into training and testing 

# Before implementing matrix factorization, we need to define the training and testing dataset. In this example the training and testing dataset is the same, with only one exception we will set some random values in the training set to zero and train the model. Then we will check how the prediction are different from the actual values.

# To use our train and test method we need a matrix with less non zeros values as possible for that we will sample a small data that includes the most rated movies and users with the most ratings

# In[279]:


# sort the dataset by movies with the most ratings
sorted_movies=ratings_movies.groupby(["movieId"])["rating"].count().reset_index()
sorted_movies=sorted_movies.sort_values('rating', ascending=False).reset_index()
sorted_movies


# In[280]:


# select the top 20 Movies with high number of ratings 
Movie_indx=sorted_movies.movieId.values
m=[]
for i in (Movie_indx[0:20]):
    m.append(i)
print(m)
most_rated_movies= ratings_movies.loc[ratings_movies['movieId'].isin(m)]


# In[219]:


# sort the dataset by users that have gave high number of  ratings
sorted_users=ratings_movies.groupby(["userId"])["rating"].count().reset_index()
sorted_users=sorted_users.sort_values('rating', ascending=False).reset_index()
sorted_users


# In[281]:


# select the top 20 users that gave high number of ratings 
Movie_indx=sorted_movies.movieId.values
m=[]
for i in (Movie_indx[0:20]):
    m.append(i)
print(m)
most_rated_movies= ratings_movies.loc[ratings_movies['movieId'].isin(m)]


# In[282]:


# Select sample data with that have high number of ratings 
user_indx=sorted_users.userId.values
n=[]
for i in (user_indx[0:20]):
    n.append(i)
print(n)
sample_data= most_rated_movies.loc[most_rated_movies['userId'].isin(n)]
sample_data


# # Convert training and testing datasets to matrices 

# In[283]:


# Create training Matrix 
T_matrix=sample_data.pivot_table(index="userId",columns="movieId",values="rating")
# Create an array and fill NA values with zeros 
T_matrix=np.array(T_matrix.fillna(0))
print(T_matrix)


# Because it takes times to run matrix factorization, i will use a small sample of sampled data dataset 

# In[284]:


# select a subset from the whole data,  we will use 20 X 20 matrix 
Test_matrix=T_matrix[:20,:20]
###
 #test and train matrix are the same matrix but later on we will set some random values in the training set to zero
 #and see how the model will predict these values 
###
Train_matrix=Test_matrix
Train_matrix


# In[285]:


# look for indexs of non-null values 
U_index,I_index=Train_matrix.nonzero()
# shuffle the indexes to get random indexes of the matrix 
np.random.shuffle(U_index)
np.random.shuffle(I_index)
print(U_index)
print(I_index)


# In[286]:


# Check the size of non null values in the original training  matrix 
print(len(U_index))
print(len(I_index))


# In[289]:


# defining the size of testing values by selecting  indexes for the values that we will set to zero 
U_index=U_index[:73]
I_index=I_index[:73]


# In[290]:


# Change values intentionaly to zero in the training set to compare how the model will predict these values 
for us,itm in zip(U_index,I_index):
    Train_matrix[us,itm]=0
Train_matrix


# In[294]:


# Define matrix factorization 
def matrix_factorization(R,K,epoch,Alpha=0.1,Beta=0.01):
    # Define matrix size 
    N=len(R) # users size 
    M=len(R[1]) # items size 
    P=np.random.rand(N,K) # initialize P matrix with random values 
    Q=np.random.rand(M,K) # initialize Q matrix with random values 
    bu=np.zeros(N) # initialize user bias as zero 
    bi=np.zeros(M) # initialize item bias as zero 
    b=np.mean(R[np.where(R != 0)]) # get ratings average 
    non_null_rating=[] # Create List to append tuples of non-null ratings maped with user and item 
    for u in (range(N)): 
        for it in (range(M)):
            if R[u,it]!=0:
                my_tuple=(u,it,R[u,it]) # tuple contains ( user, item , rating)
                non_null_rating.append(my_tuple)
    my_list=non_null_rating
    for k in (range(epoch)): # loop through iteration for gradient descent 
        np.random.shuffle(my_list) 
        for user,item,rating in (my_list):
            pred_rating=b+bu[user]+bi[item]+P[user,:].dot(Q[item,:].T) # compute initial prediction 
            error=rating-pred_rating # compute the error bestween predictions and actual values 
            #update biases 
            bu[user]+=Alpha*(error-Beta*bu[user]) # update user bias by adding learning rate 
            bi[item]+=Alpha*(error-Beta*bi[item]) # update item bias 
            #Create row copy to update matrices 
            pi=P[user,:] 
            #Update latent feature matrices 
            P[user,:]+=Alpha*(error*Q[item,:]-Beta*P[user,:]) # update P matrix 
            Q[item,:]+=Alpha*(error*pi-Beta*Q[item,:]) # update Q matrix 
        pred_matrix=b+bu[:,np.newaxis]+bi[np.newaxis,:]+P.dot(Q.T) # get the final prediction    
    return pred_matrix


# In[295]:


#run the matrix factorization to find the optimal iteration with lowest RMSE
def run_matrix(matrix):
    rmse=[] # list of RMSE 
    K_dic={} # create dictionary to map error and number of iteration 
    u_indx,i_indx=matrix.nonzero() # get indexes of non-null values 
    n=list(range(1,20)) # set number of iteration 
    for z in n:
        errors=0
        n=0
        for U,I in zip(u_indx,i_indx):
            n+=1
            pre=matrix_factorization(matrix,2,z)
            errors+=pow(matrix[U,I]-pre[U,I],2)
        RMSE=errors**0.5
        K_dic[RMSE]=z # map each iteration to RMSE value 
        rmse.append(RMSE)
    print("The Optimal Iteration:")
    print(K_dic[min(rmse)])
    print("\nThe Lowest RMSE")
    print(min(rmse)/n)
    return matrix_factorization(matrix,2,K_dic[min(rmse)])


# In[296]:


Predicted_matrix=run_matrix(Train_matrix)
Predicted_matrix


# In[297]:


#Check the RMSE of the testing data 
errors=0
N=0
for U,I in zip(U_index,I_index):# Loop through the idexes of changed values to zeros 
    N+=1 # number of changed values 
   # Compute the sum of squared errors between Prediction values and actual values 
    errors+=(Predicted_matrix[U,I]-Test_matrix[U,I])**2 
    T_SSE=errors**0.5 # sqrt of sum squared errors 
print("The Test RMSE :")
print(T_RMSE/N) # Mean sqrt of sum squared errors


# # Recommander with Surprise Package 

# Now i will try to use surprise package to predicte the ratings and compare different RMSE to the one that we computed previously using Matrix Factorization 

# In[36]:


# install scikit-surprise Package 
conda install -c conda-forge scikit-surprise


# In[27]:


# Load the necessary models ( SVD,NMF,KNN) along with other libraries 
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise import NMF
from surprise import KNNWithMeans
from surprise import Reader, Dataset


# In[115]:


# Prepare the dataset that will be user for the scikit-surprise models 
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_movies[['userId', 'movieId', 'rating']], reader)


# In[30]:


# Fit the models and use (n_folds=5 cross validation)
# svd
model_svd = SVD()
cross_validate(model_svd, data, measures=['RMSE'], cv=5, verbose=True)
# nmf
model_nmf = NMF()
cross_validate(model_nmf, data, measures=['RMSE'], cv=5, verbose=True)
# KNN 
model_knn=KNNWithMeans()
cross_validate(model_knn, data, measures=['RMSE'], cv=5, verbose=True)


# In[101]:


import matplotlib.pyplot as plt


# In[273]:


# plot the RMSE of different models
Rmse=[0.48,0.87,0.92,0.89]
models=["MF","SVD","NMF","KNN"]
plt.figure(figsize=((10,6)))
plt.plot(models, mse)
plt.xticks(models, models)
plt.xlabel("Models")
plt.ylabel("RMSE")
plt.grid(axis="y")


# From the plot we see that our  matrix factorization  has the lowest testing RMSE.

# # References :

# Recommender Systems: The Textbook 1st ed,Charu C. Aggarwal,Springer,  2016

# Hands-On Recommendation Systems with Python,Rounak Banik,  Packt Publishing Ltd, 2018

# https://en.wikipedia.org/wiki/Recommender_system

# https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf

# https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b

# https://surprise.readthedocs.io/en/stable/getting_started.html#basic-usage

# https://medium.com/@bkexcel2014/building-movie-recommender-systems-using-cosine-similarity-in-python-eff2d4e60d24
