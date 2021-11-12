import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from functools import reduce
from scipy import stats

from kmodes.kmodes import KModes
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# Feature selection - Use of Chi-squared as method of selection (Dataset consists of Categorical Variables)
def select_features(X, y):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit_transform(X_train, y_train)

    return fs

# Selecting best K for the K-Mode algorithm, Elbow method plot allow visualization of best k
def selectK (features, predictor):
    X = features
    y = predictor
    #Apply Feature Selection Function to the dataset 
    fs = select_features(X,y)
    
    #Printing Score
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
        
    #Plot Elbow Curve
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    return pyplot.show()

#Selecting Top 5 Features  
def top5_features(fs, dataset):
    
    top_5_ind = []
    for key, val in sorted(zip(fs.scores_,range(len(fs.scores_))), reverse=True):
        top_5_ind.append([key, val])
        top_5_ind = top_5_ind[:5]
        
    top_feat = {}
    top_feat_ls = []
    for i in range(len(top_5_ind)):
        top_feat[dataset.columns[top_5_ind[i][1]]] = top_5_ind[i][0]
        top_feat_ls.append(dataset.columns[top_5_ind[i][1]])
        df = dataset[top_feat_ls]

        print('Top5 Feature:{},{}'.format(dataset.columns[top_5[i][1]], top_5_ind[i][0]))
    
    return df

# Function that provides a Elbow Plot to illustrate the best K selection
def k_mode (dataset, num_k):
    cost = []
    K = range(1,num_k) # will be looping  K = 1 -> 5
    for num_clusters in list(K):
        kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, verbose=0)
        kmode.fit_predict(dataset)
        cost.append(kmode.cost_) #Cost Function for each K
    
    plt.plot(K, cost, 'bx-')
    plt.xlabel('No. of clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')
    return plt.show()

#When the optimal K is found, create a dataframe
def k_mode_compute__(dataset,num_k):
    kmode = KModes(n_clusters=num_k, init = "random", n_init = 5, verbose=0)
    clusters = kmode.fit_predict(dataset)
    a = pd.DataFrame(clusters)
    
    return a.iloc[:,0].value_counts().to_frame(), a.reset_index()
  
#Monte Carlo Simulation
def simulation(input_dataframe, num_k, num_iteration):
    freq = [] # Value_counts for n_clusters (Total No. of Samples for Each Cluster for Each iteration)
    sample = [] # 5800 Samples with label- nth_Cluster
    
    for _ in range(0,num_iteration):
        km = k_mode_compute__(input_dataframe,num_k)
        freq.append(km[0])
        sample.append(km[1])
    
    # combining Value_counts df for 100 simulation
    data_frames = freq[:]
    result = reduce(lambda df_left,df_right: pd.merge(df_left, df_right, 
                                            left_index=True, right_index=True, #on=['index'],
                                            how='outer'),data_frames)
    return freq, sample, result

#Graphical Visualization of Simulation Result
def monte_carlo_vis(simulation_result, nth_cluster):
    #Normal Distribution Tranformation
    data = simulation_result.iloc[nth_cluster].values
    
    fit_data, fit_lambda = stats.boxcox(data)
    
    mean = int(np.mean(data))
    std = int(np.std(data))
    
    area = []
    for ind, val  in enumerate(data):
        if val in range(mean-std, mean+std):
            area.append(ind)
    
    
    original = sns.distplot(fit_data, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2}, color ="green")
    
    near_mean = sns.distplot(fit_data[area], hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2}, color ="red")
    
    return original, near_mean

# Finding the Opitmal Clustering Combination Obtained from the Simulation
def optimal_cluster(simulation_result, sample):

    cl1 = []
    cl2 = []
    cl3 = []
    #Computing the Distance from the Mean (Finding the Optimal Cluster)
    for i in simulation_result.iloc[0].values:
        cl1.append(abs(i - simulation_result.iloc[0].mean()))
    for j in simulation_result.iloc[1].values:
        cl2.append(abs(j - simulation_result.iloc[1].mean()))
    for k in simulation_result.iloc[2].values:
        cl3.append(abs(k - simulation_result.iloc[2].mean()))
    
    final = zip(cl1,cl2,cl3)
    final_ = [round(sum(item),2) for item in final]
    
    #Creating a Dataframe (Opitmal Cluster)
    best_val = sample[np.where(min(final_)== final_)[0][0]][0].value_counts()
    best_sample = sample[np.where(min(final_)== final_)[0][0]]
    
    return best_val, best_sample
