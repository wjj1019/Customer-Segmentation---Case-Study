import numpy as np
from numpy import percentile
from numpy import mean
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from functools import reduce
from itertools import combinations
import itertools

def cluster_comb(dataset):
    cluster_groups = []
    # Name of Cluster___ within the dataset column
    cl_sections = sorted([i for i in dataset.columns if 'Cluster' in i])

    #Number of clusters within each cluster sections
    comb_list = [dataset[i].unique() for i in dataset.columns if 'Cluster' in i]
    comb_list = [j for i in comb_list for j in i]

    #Finding total combinations for clusters
    combinations = sorted(list(dict.fromkeys(itertools.combinations(comb_list,3))))

    for i in combinations:
        cluster_groups.append(dataset[ (dataset[cl_sections[0]] == i[0]) & (dataset[cl_sections[1]] == i[1]) & (dataset[cl_sections[2]] == i[2])])
    return cluster_groups
  
  
def select_features(X, y):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit_transform(X, y)
    return fs
  
  
#Feature selection from each cluster combination
def feat_select(dataset):    
    
    X = dataset.drop(columns=['index','Cluster','Cluster_v','Cluster_c','bichoice'])
    y = dataset['bichoice']
    
    fs = select_features(X,y)
    
    #creating a dictionary for feature selection score on each feature
    feat = {}
    for i in range(len(fs.scores_)): # Feature score is based on chi-square
        feat[i] = fs.scores_[i]
    
    #sorting the dictionary based on the value (not Key)
    # Here the Key is the Score from feature selection and Value is the location of that feature (1st ,2nd 3rd...)
    sorted_key = sorted(feat.items(),reverse=True, key=lambda kv: kv[1])    
    
    #Selecting the top 5 Features (highest score selected)
    top5 = sorted_key[:5]
    
    #From top 5, get the keys (nth feature)
    li = [] # this list contains the location of top 5 features
    for i in top5:
        li.append(i[0])
    
    feat_s = X[X.columns[li]].reset_index() #Selecting the nth feature from the X dataframe 
    bi = y.reset_index()
    
    clustering_comb = pd.merge(feat_s,bi) #Merge with the Outcome an return the dataframe having Top5 features + Outcome
    
    return clustering_comb
  
def looping (cluster_group):
    #looping all the cluster (27) select feature and retun the dataframe with only top 5 features (total output number is 27)
    cluster_comb = [] #This list will contain the Dataframe obtained from feat_select function for all 27 combinations
    for i in cluster_group:
        cluster_comb.append(feat_select(i))
    return cluster_comb

# Getting the probabilities Matrix used for Naive Bayes formula 
def probability (dataset): #Dataframe inserted here will be the cluster_comb[:]
    
    li = []
    # Loop through columns obtained from cluster_comb list (above)
    for features in dataset.columns[1:-1]: # Top 5 Feature names selected
        for i in dataset[features].value_counts().keys(): #get the classes within each feature(Ex, Range: 100 200 300 400)
            
            #Ex. P(Xn|Y) = P(Range = 30,000 | Bichoice = 1) = Total # of Range = 30,000 where bichoice = 1/ Total # bichoice = 1
            #Ex. For Cluster_comb[0], that is Cluster 0-0-0, The Total Sample Size is 751 -> 400 for y=1 and 351 for y=0
            # These values witll be the Denominator, Numerator will be depending on the Specific Class with a certain Feature
            p = dataset[dataset[features] == i]['bichoice'].value_counts()/dataset['bichoice'].value_counts()
            p = p.to_frame().reset_index()
            p.rename(columns={'bichoice': '{} {}'.format(features,i) },inplace=True)
            li.append(p)

    data_frames = li[:]
    result = reduce(lambda df_left,df_right: pd.merge(df_left, df_right, 
                                              left_index=True, on=['index'],#right_index=True,
                                              how='outer'),data_frames)
    result.drop(['index'],axis=1,inplace=True)
    result.fillna(0,inplace=True)
    
    return result

  
# Dataframe: one out of 27 Cluster combination
# Prior: The percentange calculated from the Dataframe (1/27 Cluster comb)
# y: Defining whether we would like to see the probabilities of bichoce 0 or 1
def Naive_bayes(dataset,likelihood_matrix,y):
    
    numerator = [] #bichoice = 0: P(Xn|Y=0).... *P(Y=0)
    denominator = []# P(Xn|not y = 1)P(not y =1)
    
    for i in range(len(dataset)):
        likelihood = 1
        likelihood_not = 1
        for key,val in dict(dataset.iloc[i,1:-1]).items():
            match = '{} {}'.format(key,val)
            if match in likelihood_matrix.columns:
                
                probability = likelihood_matrix[match][y]
                likelihood = likelihood * probability
                prior = len(dataset[dataset.bichoice == y]/ len(dataset))
                
                pxn_ynot = likelihood_matrix[match][(y -1) * -1]
                likelihood_not = likelihood_not * pxn_ynot
                prior_not = len(dataset[dataset.bichoice == (y-1) * -1]/ len(dataset))
            
        numerator.append(likelihood * prior)
        denominator.append(likelihood_not*prior_not)

    # P(Xn|y)p(y) + P(Xn|not y)P(not y) = Denominator (evidence)
    denominator = [(i+j) for i,j in zip(numerator,denominator)] 

    total = [int((numer/denomi)*100) for numer,denomi in zip(numerator, denominator)] #Numerator/ Denominator
    total = [0 if i ==100 else i for i in total]
    
    return total
  
def nb_loop(cluster_comb):
    #Running all of 27 cluster combination with generated functions to get the final result as probability 
    bayes_proba = []# The list contains all the probabilities calculated for each samples in each 27 clusters combination 
    for i in cluster_comb: #27 Combination in a list format
        pr = probability(i)
        bayes_proba.append(Naive_bayes(i,pr,0)) # Compute Naive Bayes Algorithm and add to the list 
    return bayes_proba 

  
# Finding the Minimum and Maximum Probability obtained from each Cluster
def min_max (nb_prob):
    min_max = []
    for cluster_comb in nb_prob:
        mean = np.mean(cluster_comb)
        for sample in range(len(cluster_comb)):
            if cluster_comb[sample] == 0:
                cluster_comb[sample] = mean
        min_max.append([min(cluster_comb),np.where(np.array(cluster_comb)==min(cluster_comb))[0][0],
                                       max(cluster_comb),np.where(np.array(cluster_comb)==max(cluster_comb))[0][0]])
    return min_max

  
def prob_seperation(nb_prob, cluster_comb):
    high = [] #High probability of buychoice being 0 -> Not willing to buy Electric vehicle with given conditions
    low = [] #Lower probability of buychoice being 0 -> Higher willingness to buy Electric vehicles with given conditions 
    for i in range(len(nb_prob)):
            high.append([index for index, prob in enumerate(nb_prob[i]) if prob > 75])
            low.append([index for index, prob in enumerate(nb_prob[i]) if prob < 25])
            
    high_group = [] # For Each Cluster, High Probability of Bichoice 0 is gathered -> Meaning Low probability of bichoice 1
    lower_group = [] # For Each Cluster, Low Probability of Bichoice 0 is gathered -> Meaning High Probability of bichoice 1
    for i in range(len(high)):
        a = high[i]
        high_group.append(cluster_comb[i].iloc[[*a]])

    for j in range(len(low)):
        b = low[j]
        lower_group.append(cluster_comb[j].iloc[[*b]])
    
    return high_group, lower_group
    
