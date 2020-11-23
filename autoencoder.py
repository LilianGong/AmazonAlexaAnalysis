import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
import clustering as c

from time import time

from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import VarianceScaling

import fpath as f
import processing as p

seed = 888
rng = np.random.RandomState(seed)


'''data preparation'''
def get_train_xy(df,indexes,paras):
    train_x = df[paras].values
    train_y = df[indexes]
    return train_x,train_y


'''model training'''
def train_model(train_x,mname,nntrain="True",d = 200):
    # input placeholder
    input_embedding = Input(shape=(train_x.shape[1],))
    # encoding embedding data
    encoded = Dense(d, activation='relu')(input_embedding)
    encoded = Dense(round(d/2), activation='relu')(encoded)
    encoded = Dense(round(d/4), activation='sigmoid')(encoded)
    # reconstruction of the data
    decoded = Dense(round(d/2), activation='relu')(encoded)
    decoded = Dense(d, activation='relu')(decoded)
    decoded = Dense(train_x.shape[1])(decoded)

    if nntrain == "True":
    # maps input to its reconstruction
        autoencoder = Model(input_embedding, decoded)
        #  maps input to its encoded representation
        autoencoder.compile(optimizer='adam', loss='mse')

        train_history = autoencoder.fit(train_x,
                                    train_x,
                                    epochs=500,
                                    batch_size=2048)
        hist_df = pd.DataFrame(train_history.history)
        hist_df.to_csv(os.path.join(f.nnmodel,"history_{}_{}.csv".format(dsize,round(time()))))
        autoencoder.save(os.path.join(f.nnmodel, mname))
    else:
        autoencoder = load_model(os.path.join(f.nnmodel, mname))
    encoder = Model(input_embedding, encoded)
    return autoencoder, encoder


'''predict result'''
def predict_result(train_x,encoder):
    return encoder.predict(train_x)


if __name__ == "__main__":
    '''prep data'''
    embedding_df = pd.read_csv(os.path.join(f.nntrain,"trainset_w_meanfill.csv"))
    paras = [c for c in embedding_df.columns[2:] if "kmeans" not in c]
    train_x = embedding_df[paras]
    train_x = pd.DataFrame(p.scale(train_x))
    print(train_x.head())

    '''training'''

    training = "False"
    dsize =  380
    if training == "True":
        print("now trainning autoencoder with the structure : {}-{}-{}-{}-{}-{}-{}.".\
              format(train_x.shape[1],
                     dsize,
                     round(dsize/2),
                     round(dsize/4),
                     round(dsize/2),
                     round(dsize),
                     train_x.shape[1]))
    else:
        print("now reading nn model with dsize : {}".format(dsize))

    autoencoder, encoder = train_model(train_x,"raw_embed_dsize{}".format(dsize),training,dsize)
    nn_df = pd.DataFrame(predict_result(train_x,encoder))
    nn_df.columns = ["para_"+str(c) for c in nn_df.columns]
    print("nn result calculated!")

    '''clustering'''
    print("start clustering...")
    kmin = 13
    kmax = 14
    inter = 1
    result = c.multi_clustering(nn_df,kmin,kmax,inter)
    nn_df[result.columns] = result
    nn_df.to_csv(os.path.join(f.nn_result,"raw_embed_dsize{}_k{}-{}.csv".format(dsize,kmin,kmax)), index=False)
    print("clustering successful! k ranges from {} to {}".format(kmin,kmax))






