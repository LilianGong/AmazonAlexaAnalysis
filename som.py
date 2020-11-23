import pandas as pd
import SimpSOM as sps
import numpy as np
from time import time
import matplotlib.pyplot as plt
import clustering as c
import processing as p
from argparse import ArgumentParser
import os
import fpath as f
from tqdm import tqdm
import re
import pca


parser = ArgumentParser(description ='Self-organized Map')
parser.add_argument('--train', default = "True",
                    help = 're-train the module or not',
                    type = str)
args = parser.parse_args()


def add_meta_data(tartan_corpus,df):
    conv_meta_table = tartan_corpus.get_conversations_dataframe(). \
        reset_index(). \
        rename(columns={'id': 'conversationId'})
    conv_meta_table = p.get_meta(conv_meta_table)
    df = df.merge(conv_meta_table)
    print(df.head())
    print("meta data added")
    df.to_csv(os.path.join(f.aggdata_dir,"som_train_set_{}.csv".format(round(time()))),index = False)
    return df


def get_mapping_columns(colname):
    mapping = ["emotion","next_topic","greeting","plan"]
    a = r"([a-zA-Z]+)([0-9]+)"
    return [col for col in colname if re.search(a,col) != None]


def train_som(df, height=20,width=20):
    start = time()
    print(f.sommodel)
    net = sps.somNet(height, width, df, PCI=True, PBC=False)
    net.train(0.01, 20000)
    net.save(os.path.join(f.sommodel,"som_weights_{}_{}_{}".format(height,width,round(time()))))
    # net.nodes_graph(colnum=0)
    end = time()
    print("som training time : ".format(end-start))
    return net


def get_som_dat(train,net):
    print("start projecting...")
    start = time()
    prj = np.array(net.project(train))
    end = time()
    print("Succeed! projection time : {}".format(end-start))
    plt.scatter(prj.T[0],prj.T[1])
    plt.show()
    return prj


def som_mean_dis(dat,net):
    start = time()
    dis_ls = []
    # bmu_arr = np.array([])
    for vec in tqdm(dat):
        bmu = net.find_bmu(vec)
        dis = np.sqrt(np.sum((vec-bmu.weights)**2))
        dis_ls.append(dis)
        # bmu_arr = np.append(bmu_arr,bmu.weights)
    mean_dis = np.mean(dis_ls)
    end = time()
    print(mean_dis)
    print("Succeed! calculate mean distance time : {}".format(end - start))
    return mean_dis


def som_cluster(prj_dat,k_min=2,k_max=10,inter = 1):
    res = c.multi_clustering(prj_dat, k_min, k_max,inter)
    return res



if __name__ == "__main__":

    '''prepare trainning data - fill na'''
    # use all 400-dimension embedding data
    # tartan_corpus, conv_ids, speaker_ids = p.load_corpus(f.corpus_dir)
    # embedding_df = p.load_expanded_embedding_df(os.path.join(f.aggdata_dir,"embedding_all.csv"),fill_mean = True)
    # embedding_df = add_meta_data(tartan_corpus,embedding_df)
    pca_process = False
    if pca_process == True:
        print("pca starting...")
        embedding_df = pd.read_csv(os.path.join(f.aggdata_dir,"embedding_all.csv"))
        map_keys = ["emotion", "next_topic", "greeting", "plan"]
        for para in map_keys:
            embedding_df[para] = embedding_df[para]. \
                apply(lambda x: [float(a) for a in x.replace("\n", ""). \
                      strip(']['). \
                      split(" ") if a != ""])

        indexes = embedding_df.columns[:2]
        embedding_df = pca.get_pca_df(embedding_df, embedding_df.columns[:2], map_keys, 0.8, "True")


    else:
        embedding_df = pd.read_csv(os.path.join(f.nntrain,"trainset_w_meanfill.csv"))


    paras = [c for c in embedding_df.columns[2:] if "kmeans" not in c]
    mapping_paras = get_mapping_columns(embedding_df.columns)
    '''scaling'''
    train = embedding_df[paras].values
    train = p.scale(train)
    print(train.shape)

    # '''test set'''
    # train = train[np.random.randint(train.shape[0], size=1000), :]
    print(train.shape)

    if args.train == "True":
        '''train SOM'''
        som = train_som(train,20,20)

    else:
        '''load trained weights'''
        fpath = "som_weights_20_20_1600831821.npy"
        som = sps.somNet(20,20,train,loadFile=os.path.join(f.sommodel,"som_weights_20_20_1600831821.npy"))
        print("som model loaded")

    '''evaluate mean error'''

    # me = som_mean_dis(train, som)
    #
    # print("mean distance : {}".format(me))
    #
    #
    # '''get SOM projection'''
    # prj = pd.DataFrame(get_som_dat(train, som))
    # prj.to_csv(os.path.join(f.som_result,"som_data_{}.csv".format(round(time()))),index = False)

    '''read SOM projection'''
    prj = pd.read_csv(os.path.join(f.som_result,"som_data_pca_1600956509.csv"))
    prj.columns = ["para1", "para2"]

    # '''cluster'''
    kmin = 13
    kmax = 14
    inter = 1
    cluster_res = som_cluster(prj,kmin,kmax,inter)
    som_cluster = pd.concat([embedding_df[paras],cluster_res],axis = 1)
    som_cluster.to_csv(os.path.join(f.som_result,"som_cluster_pca_embedding_{}_{}.csv".format(kmin,kmax)),index=False)

