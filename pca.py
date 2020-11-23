from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


#return pca result
def fit_pca(data,para):
    pca = PCA()
    X_train = pd.DataFrame.from_records(data[para].values)
    X_new = pca.fit_transform(X_train)
    return pca, X_new


#plot scree plot of oca
def plot_pca(pca):
    variance_ratio = pca.explained_variance_ratio_
    cumu_variance_ratio = [sum(variance_ratio[:i]) for i in range(1,len(variance_ratio))]
    # fig1 = plt.plot(variance_ratio,color = '#ed6663')
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # plt.xticks(rotation=0)
    # plt.xlabel("principle component")
    # plt.ylabel("variance ratio")
    # plt.title("scree plot of variance")
    # plt.show()
    # plt.savefig("scree_plot.png")
    #
    # fig2 = plt.plot(cumu_variance_ratio,color = '#0f4c81')
    # plt.xticks(rotation=0)
    # plt.xlabel("principle component")
    # plt.ylabel("cumulative variance ratio")
    # plt.title("cumulative variance ratio")
    # plt.savefig("cumu_variance.png")
    # plt.show()
    return variance_ratio,cumu_variance_ratio


#find num of pcas to retain certain variance
def find_retain(cumu_variance_ratio,threshold):
    for ratio in cumu_variance_ratio:
        if ratio >= threshold:
            w = cumu_variance_ratio.index(ratio)+1
            print(str("{} principal components should be retained "\
            .format(cumu_variance_ratio.index(ratio)+1) +
             "to capture at least {} of the total variance.".format(threshold)))
            return w


#reconstruction error calculation
def reconstruction_err(retain,variance_ratio):
    w = str("{} is the reconstruction error "\
    .format(round(sum(variance_ratio[retain:]),3))
    +"if we only retain top {} of the PCA components.".format(retain))
    print(w)


#get pca threshold of certain para
def get_pca_threshold(embedding_df,para,t,retain):
    print("parameter : {}".format(para))
    pca,X_new = fit_pca(embedding_df,para)
    v_ratio, cumu_v_ratio = plot_pca(pca)
    w = find_retain(cumu_v_ratio ,t)
    print("for {} : ".format(para))
    reconstruction_err(retain,v_ratio)


#get pca df
def get_pca_df(embedding_df,mapping,t):
    pca_df = embedding_df[["unique_index","gender","age_group"]]
#     para_count = 0
    for para in mapping.keys():
        pca,X_new = fit_pca(embedding_df,para)
        v_ratio, cumu_v_ratio = plot_pca(pca)
        w = find_retain(cumu_v_ratio ,t)
        para_df = pd.DataFrame(X_new).iloc[:,:w]
        para_df.columns = [para+"_"+str(i) for i in range(w)]
        pca_df = pd.concat([pca_df,para_df],axis = 1)
#         para_count += w
    return pca_df


