import numpy as np
import pandas as pd
from time import time

def helper_Clist(rawList):
    if all([isinstance(raw,pd.DataFrame) for raw in rawList]):
        return [raw.values for raw in rawList]
    elif all([isinstance(raw,np.ndarray) for raw in rawList]):
        return rawList



# this calculates the maximum distance of the 2 points inside cluster i
def intracluster_distance(Ci):
    a = np.arange(len(Ci))
    intra_dis = np.zeros((len(Ci),len(Ci)))
    mesh = np.array(np.meshgrid(a,a))
    iters = mesh.T.reshape(-1,2)
    for i, j in iters:
        intra_dis[i,j] = np.linalg.norm(Ci[i]-Ci[j])
    return np.max(intra_dis)


# this calculates the minimum distance between cluster i & cluster j
def intercluster_distance(Ci, Cj):
    inter_dis = np.full((len(Ci),len(Cj)),np.inf)
    a = np.arange(len(Ci))
    b = np.arange(len(Cj))
    mesh = np.array(np.meshgrid(a,b))
    iters = mesh.T.reshape(-1, 2)
    for i, j in iters:
        inter_dis[i,j] = np.linalg.norm(Ci[i]-Cj[j])
    return np.min(inter_dis)



def calculate_dunn(Ci,Cj):
    intra = intracluster_distance(Ci)
    inter = intercluster_distance(Ci, Cj)
    dunn = inter/intra
    return dunn


# get dunn index
def dunn_index(Clist):
    start = time()
    Clist = helper_Clist(Clist)
    dunn_index = np.inf
    iters = len(Clist)

    for i in range(iters):
        for j in range(iters):
            if i<j:
                dunn = calculate_dunn(Clist[i],Clist[j])
                print("calculated dunn for cluster {} and cluster {} is : {}".\
                      format(i,j,dunn))
                if dunn < dunn_index:
                    dunn_index = dunn
    end = time()
    print("calculation time : {}".format(end-start))
    return dunn_index


def get_dunn_index(df,k_cluster):
    cluster_list = []
    for k in range(k_cluster):
        selected_df = df[df['kmeans_'+str(k_cluster)]==k]
        select_columns = [c for c in selected_df.columns if "kmeans" not in c]
        cluster_list.append(selected_df[select_columns].values)
    dunn = dunn_index(cluster_list)
    return dunn


def compare_dunn_index(df,k_max):
    indexes = {}
    for k in range(2,k_max+1):
        print("now calculating k = {} cluster".format(k))
        dunn_index = get_dunn_index(df,k)
        print("dunn index is {}".format(dunn_index))
        indexes[k] = dunn_index
    indexes_df = pd.DataFrame.from_dict(indexes.items())
    indexes_df.columns = ['num_of_cluster','dunn_index']
    return indexes_df


def entropy(df,k_cluster):
    df['counter'] = 1
    distribution = pd.DataFrame(df.groupby('kmeans_'+str(k_cluster)).sum()['counter'].reset_index())
    entropy = np.sum([-c/len(df)*np.log(c/len(df)) for c in distribution['counter']])
    return entropy


def compare_entropy(df,k_max):
    entropys = {}
    for k in range(2,k_max+1):
        ent = entropy(df,k)
        entropys[k] = ent
    entropys_df = pd.DataFrame.from_dict(entropys.items())
    entropys_df.columns = ['num_of_cluster', 'entropy']
    return entropys_df


def clumpiness(df,k_cluster):
    df['counter'] = 1
    distribution = pd.DataFrame(df.groupby('kmeans_' + str(k_cluster)).sum()['counter'].reset_index())
    pen = (len(df)-np.max(distribution['counter']))/len(df)
    return pen


def compare_clumpiness(df,k_max):
    clumps = {}
    for k in range(2,k_max+1):
        clump = clumpiness(df,k)
        clumps[k] = clump
    clumpiness_df = pd.DataFrame.from_dict(clumps.items())
    clumpiness_df.columns = ['num_of_cluster', 'clumpiness']
    return clumpiness_df

def clumpiness_min(df,k_cluster):
    df['counter'] = 1
    distribution = pd.DataFrame(df.groupby('kmeans_' + str(k_cluster)).sum()['counter'].reset_index())
    pen = (len(df)-np.min(distribution['counter']))/len(df)
    return pen

def compare_clumpiness_min(df,k_max):
    clumps = {}
    for k in range(2,k_max+1):
        clump = clumpiness_min(df,k)
        clumps[k] = clump
    clumpiness_df = pd.DataFrame.from_dict(clumps.items())
    clumpiness_df.columns = ['num_of_cluster', 'clumpiness']
    return clumpiness_df

if __name__ == "__main__":
    pca_cluster = pd.read_csv("aggdata/pca_cluster.csv")
    Clist_k3 = [pca_cluster[pca_cluster['kmeans_3']==0].iloc[:,3:45],
                pca_cluster[pca_cluster['kmeans_3']==1].iloc[:,3:45],
                pca_cluster[pca_cluster['kmeans_3']==2].iloc[:,3:45]]


    testCi = np.array([[0,0,0],[1,1,1],[2,2,2]])
    testCj = np.array([[2,0,1],[1,1,10],[2,0,2]])
    testClist = [testCi,testCj]
    # print(intracluster_distance(testCi))
    # print(intercluster_distance(testCi,testCj))
    # print(dunn_index(testClist))
    # print("final dunn index : {}".format(dunn_index(Clist_k3)))
    # start = time()
    # print("final dunn index : {}".format(get_dunn_index(pca_cluster.iloc[:, 3:45], 3)))
    # end = time()
    # print("calculation time : {}".format(end - start))
    # print(compare_entropy(pca_cluster, 10))
    # print(compare_clumpiness(pca_cluster, 10))
    print(compare_clumpiness_min(pca_cluster, 10))
