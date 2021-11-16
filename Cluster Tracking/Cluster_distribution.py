import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns


def cluster_dist(initial_cluster, num_second_cluster, dataset):
    cluster_dist = []
    second_cluster = [i for i in range(0, num_second_cluster)]
    cluster_sections = sorted([i for i in dataset.columns if 'Cluster' in i])
    
    # Looping though the Cluster of Vehicle Usage 
    for nth_cluster in second_cluster:
        cluster_dist.append(dataset[(dataset[cluster_sections[0]] == initial_cluster) & (dataset[cluster_sections[1]] == nth_cluster)]
                         [cluster_sections[-1]].value_counts().to_frame().reset_index())
        
        #From Cluster 0 -> Cluster 0 -> Cluster_C Value counts to determine how many samples traveled
    
    #Mering the dataframe of Cluster Combination
    data_frames = cluster_dist[:]
    result = reduce(lambda df_left,df_right: pd.merge(df_left, df_right, 
                                              left_index=True, right_index=True, on=['index'],
                                              how='outer'),data_frames)
    
    result.columns = [cluster_sections[1],
                      cluster_sections[-1]+'{}'.format(1), 
                      cluster_sections[-1]+'{}'.format(2),
                      cluster_sections[-1]+'{}'.format(3)]
    
#     result = result.sort_values(by=cluster_sections[1])
    result['Cluster_PI'] = initial_cluster
    
    return result
  

  
def pivot_visualize(dataset):
    clusters = []
    cluster_sections = [i for i in dataset.columns if 'Cluster' in i]
    cl_length = dataset[cluster_sections[0]].nunique()
    total_sample = [len(dataset[dataset[cluster_sections[0]] == i]) for i in range(cl_length)]

    for i,j in zip(range(cl_length),total_sample):
        cl = cluster_dist(i,3,dataset)
        cl.iloc[:, 1:4] = cl.iloc[:, 1:4]/j *100
        
        clusters.append(cl)
        combined = pd.concat(clusters)
    
    pivot = pd.pivot_table(combined, index = ['Cluster_PI', 'Cluster_c'])
        

    return pivot
 

def mode_selection (cluster_df, nth_cluster):
    mode = {}
    for i in cluster_df.columns:
        mode_val = cluster_df[i].value_counts()[:1].index.values
        mode[str(i)] = mode_val
        print('Cluster {} Feature:{}, Mode: {}'.format(nth_cluster, i, cluster_df[i].value_counts()[:1].index.values))
    return mode
