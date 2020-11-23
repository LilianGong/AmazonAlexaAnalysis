from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clustering(X_train,k_cluster):
    kmeans = KMeans(n_clusters=k_cluster,random_state=8).fit(X_train.values)
    return kmeans.labels_


# construct multiple cluster at once
def multi_clustering(df,k_max):
    for i in range(2,k_max+1):
        res = clustering(df, i)
        df['kmeans_'+str(i)] = res
    return df


def concat_ls(x, mapping):
    _all = []
    for k in mapping.keys():
        _all.extend(x[k])
    return np.array(_all)


def get_distribution_plot(df,num_cluster,gp_elm,col):
    df['counter'] = 1
    gp_elm.insert(0, "kmeans_"+str(num_cluster))
    if len(gp_elm) == 2:
        bar_df = pd.DataFrame(df.drop_duplicates('unique_index').\
                              groupby(gp_elm).count()['counter']).unstack()
        bar_df = bar_df['counter']
    else:
        bar_df = pd.DataFrame(df.drop_duplicates('unique_index').\
                          groupby(gp_elm).count()['counter']).unstack(-1).unstack(1)
        bar_df = bar_df['counter']
        bar_df.columns = ['_'.join(col).strip() for col in bar_df.columns.values]
    print(bar_df)
    res = bar_df.div(bar_df.sum(axis=1), axis=0)
    res.plot.bar(color=col,
                 stacked=True,
                 rot=1,
                 figsize=(8,5))
    plt.title(",".join(gp_elm[1:])+" distribution")
    plt.ylabel("count")
    plt.xlabel("group")
    plt.show()

    