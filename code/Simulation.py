import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" In the following I define all the functions needed for the simulation  """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""This function should be use for the simulation based on COSINUS-SIMILARITY """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def create_cossim_matrix(data):
    multi = data.set_index(['userId', 'movieId'])['rating']
    matrix = multi.unstack().T.fillna(0)
    cos_sim_matrix=cosine_similarity(matrix.T)
    cos_sim_matrix=pd.DataFrame(cos_sim_matrix,columns=set(data['userId']),index=set(data['userId']))
    return cos_sim_matrix

def create_closest_user_table(cos_sim_matrix,k=6):
    k_closest_users=[] 
    for i in range(1,cos_sim_matrix.shape[0]+1):
        most_sim_=cos_sim_matrix.sort_values(by=i,ascending=False)[i][0:k].index
        k_closest_users.append(most_sim_)
    k_closest_users=pd.DataFrame(k_closest_users,index=cos_sim_matrix.columns)
    k_closest_users=k_closest_users.T.drop(0,axis=0)    
    return k_closest_users

def create_favourites_table(data,m=20):
    m_favourite_moviesId=[] 
    for i in set(data['userId']):
        m_favourites_=list(data[data['userId']==i].sort_values(by='rating',ascending=False)['movieId'][0:m])
        m_favourite_moviesId.append(m_favourites_)
    m_favourite_moviesId=pd.DataFrame(m_favourite_moviesId,index=set(data['userId'])).T
    return m_favourite_moviesId

def create_recommender_table(m_favourite_moviesId,k_closest_users,k=3,su=5): # k = number of bestrated movies per similiar user; su = number of most similiar users
    recommender_table=[]
    for i in range(1,k_closest_users.shape[1]+1):
        for j in range(1,su+1):
            recommender_table.append(m_favourite_moviesId[k_closest_users[i][j]][0:k])
    recommender_table=pd.DataFrame(np.array(recommender_table).reshape(k_closest_users.shape[1],su*k).T,columns=k_closest_users.columns)
    return recommender_table

def pick_check_and_rate_movie(user,data,recommender_table,k=3,su=5,exp=1):
    check=0
    counter=0    
    while check==0:  # check=0 means, that there is not entry yet
        next_movie=np.random.choice(recommender_table[user],p=picking_probabilities(k,su,exp))
        rating=data[data['movieId']==next_movie]['rating'].mean()
        check=data[(data['userId']==user) & (data['movieId']==next_movie)].shape[0]
        counter+=1
        if check==1 and counter<5:    
            check=0
            #print(f'{i}: user {user} has watched movie. Draw again.')
        elif check==1 and counter==5:
            #print(f'{i}: user {user} has watched all movies. No more movies are drawn.')
            return next_movie, rating, counter        
        else:
            counter=0
            return next_movie, rating, counter

"""These functions should be use for the simulation based on WEIGHTED RANKING """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def calculate_weighted_rank(data,m=100):
    v = data['no_of_ratings']
    R = data['av_rating']
    C = data['rating'].mean()
    def weighted_rank(R, v, C, m):
        return R * v / (v+m) + C * m / (v+m)
    data['weighted_rank']=weighted_rank(R,v,C,m) 
    return data

def create_chartlis(data):
    chartlist=data.sort_values('weighted_rank',ascending=False).drop_duplicates('movieId')['movieId']
    return chartlist

def pick_and_rate_movie_simple(user,data,recommender_table,k=3,su=5,exp=1):
    check=0
    counter=0
    recommender_table=recommender_table[0:k*su]
    while check==0:  # check=0 means, that there is not entry yet
        next_movie=np.random.choice(recommender_table,p=picking_probabilities(k,su,exp))
        rating=data[data['movieId']==next_movie]['rating'].mean()
        check=data[(data['userId']==user) & (data['movieId']==next_movie)].shape[0]
        counter+=1
        if check==1 and counter<5:     
            check=0
            #print(f'{i}: user {user} has watched movie. Draw again.')
        elif check==1 and counter==5:
            #print(f'{i}: user {user} has watched all movies. No more movies are drawn.')
            return next_movie, rating, counter        
        else:
            counter=0
            return next_movie, rating, counter
 
"""           These functions are needeed for either simulation            """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def picking_probabilities(k=3,su=5,exp=1): # exp: power for prob -> more prob for higher ranking
    if exp==1:
        probabilities=np.array(range(1,k*su+1))
        probabilities=probabilities/(sum(probabilities))
    if exp==2:
        probabilities=np.array(range(1,k*su+1))**2
        probabilities=probabilities/(sum(probabilities))
    return probabilities[::-1]

def create_entry(user,next_movie,rating,data):
    new_entry=pd.DataFrame([[user,next_movie,rating]],columns=['userId','movieId','rating'])
    data=data.append(new_entry)
    return data              # maybe here is the slow-source
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""                   SIMULATOIN FOR COSINUS-SIMILARITY                    """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
data = pd.read_csv('ml-latest-small/movie_data.csv')
u=610 # number of user IDs from 1 to u
r=10 # number of simulation rounds

for j in range(1,r+1):
    print(j)
    for i in list(range(1,u+1))*j:
        user=i
        k=4
        su=4
        cosim_matrix=create_cossim_matrix(data)
        closest_user_table=create_closest_user_table(cosim_matrix)
        favourires_table=create_favourites_table(data)
        recommender_table=create_recommender_table(favourires_table,closest_user_table,k,su)
        next_movie,rating,counter=pick_check_and_rate_movie(user,data,recommender_table,k,su)
        if counter==5:
            continue
        data=create_entry(user,next_movie,rating,data)
        #print(f'{i}: dataset updated')
data.to_csv('results/simulated_data_COSINUS-SIMILARITY.csv', sep=',', encoding='utf-8')
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""                   SIMULATOIN FOR WEIGHTED-RANKING                      """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
data = pd.read_csv('ml-latest-small/movie_data.csv')
u=610 # number of user IDs from 1 to u
r=10 # number of simulation rounds

for j in range(1,r+1):
    print(k)
    for i in list(range(1,u+1))*j:
        user=i
        k=5
        su=5
        chartlist=create_chartlis(data)
        next_movie,rating,counter=pick_and_rate_movie_simple(user,data,chartlist,k,su,exp=1)
        if counter==5:
            continue
        data=create_entry(user,next_movie,rating,data)
        #print(f'{i}: dataset updated')
data.to_csv('results/simulated_data_WEIGHTED-RANKING.csv', sep=',', encoding='utf-8')
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""





