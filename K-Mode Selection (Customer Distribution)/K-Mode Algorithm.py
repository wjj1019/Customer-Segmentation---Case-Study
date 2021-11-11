import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from kmodes.kmodes import KModes
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# Feature selection - Use of Chi-squared as method of selection (Dataset consists of Categorical Variables)
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

# Selecting best K for the K-Mode algorithm, Elbow method plot allow visualization of best k
def selectK (features, predictor):
    X = features
    y = predictor
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
    
    #Apply Feature Selection Function to the dataset 
    X_train_fs, X_test_fs, fs = select_features(X_train,y_train, X_test)
    
    #Printing Score
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
        
    #Plot Elbow Curve
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    return pyplot.show()
  
# Function that provides a Elbow Plot to illustrate the best K selection
def k_mode (dataset, num_k):
    cost = []
    K = range(1,num_k) # will be looping  K = 1 -> 5
    for num_clusters in list(K):
        kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, verbose=1)
        kmode.fit_predict(dataset)
        cost.append(kmode.cost_) #Cost Function for each K
    
    plt.plot(K, cost, 'bx-')
    plt.xlabel('No. of clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')
    return plt.show()
  

#When the optimal K is found, create a dataframe
def k_mode_compute__(dataset,num_k):
    kmode = KModes(n_clusters=numk, init = "random", n_init = 5, verbose=1)
    clusters = kmode.fit_predict(dataset)
    a = pd.DataFrame(clusters)
    return a.iloc[:,0].value_counts().to_frame(), a.reset_index()
  
  
  #Simulation for 100 times - Averaging the 100 Simulation K-Mode Algorithm's Selection
  def simulation(section, num_k):
    freq = [] # Value_counts for 3 clusters resulted from given dataframe (Question Sections)
    sample = [] # 5800 Samples corresponds to the value_counts df location
    
    for _ in range(0,100):
        km = k_mode_compute__(section,num_k)
        freq.append(km[0])
        sample.append(km[1])
    
    # combining Value_counts df for 100 simulation
    data_frames = freq[:]
    result = reduce(lambda df_left,df_right: pd.merge(df_left, df_right, 
                                            left_index=True, right_index=True, #on=['index'],
                                            how='outer'),data_frames)
    
    cl1 = []
    cl2 = []
    cl3 = []
    #Computing the Distance from the Mean (Finding the Optimal Cluster)
    for i in result.iloc[0].values:
        cl1.append(abs(i - result.iloc[0].mean()))
    for j in result.iloc[1].values:
        cl2.append(abs(j - result.iloc[1].mean()))
    for k in result.iloc[2].values:
        cl3.append(abs(k - result.iloc[2].mean()))
    
    final = zip(cl1,cl2,cl3)
    final_ = [round(sum(item),2) for item in final]
    
    #Creating a Dataframe (Opitmal Cluster)
    best_val = sample[np.where(min(final_)== final_)[0][0]][0].value_counts()
    best_sample = sample[np.where(min(final_)== final_)[0][0]]
    
    return best_val, best_sample
