import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import sys
sys.path.append("Fast_Sentence_Embeddings/fse")
import warnings
warnings.filterwarnings("ignore")


'''create embedding map parameter'''
def get_mapping(utt_table):
    b_utt = pd.unique(utt_table['bot'].dropna())
    emotion = [b for b in b_utt if 'How are you' in b]
    next_topic = [b for b in b_utt if 'talk about' in b \
                  or 'interested' in b or 'chat about?' in b or 'topics' in b]
    greeting = [b for b in b_utt if 'Hello there' in b \
                or 'Glad to talk to you' in b]
    plans = [b for b in b_utt if 'plan' in b \
             or 'What are you doing' in b]
    # name = [b for b in b_utt if 'your name' in b]
    mapping = {'emotion':emotion,
               'next_topic':next_topic,
               'greeting':greeting,
               'plan':plans}
    #           'name':name}
    return mapping


'''create pre embedding df'''
def prompt_clasify(utt_table, mapping):
    def get_key(x):
        for k in mapping.keys():
            if x in mapping[k]:
                return k
        return None

    utt_table['b_utt_gp'] = utt_table['bot'].apply(lambda x: get_key(x))
    valid_utt_table = utt_table.dropna(subset=['b_utt_gp'])
    prep_embedding_df = valid_utt_table.pivot_table(index=['unique_index', 'gender', 'age_group'],
                                                    columns='b_utt_gp',
                                                    values='user',
                                                    aggfunc='first') \
        .reset_index() \
        .rename_axis(None, axis=1)
    return prep_embedding_df


'''sif imbeddings'''
def sif_embeddings(s, mpath, alpha=1e-3):
    REAL = np.float32
    model = Word2Vec.load(mpath)
    vlookup = model.wv.vocab  # Gives us access to word index and count
    vectors = model.wv        # Gives us access to word vectors
    size = model.vector_size  # Embedding size
    Z = 0
    for k in vlookup:
        Z += vlookup[k].count # Compute the normalization constant Z

    count = 0
    v = np.zeros(size, dtype=REAL) # Summary vector
    for w in s:
        if w in vlookup:
            for i in range(size):
                v[i] += ( alpha / (alpha + (vlookup[w].count / Z))) * vectors[w][i]
            count += 1
    if count > 0:
        for i in range(size):
            v[i] *= 1/count
    return v


def get_embedding_df(prep_embedding_df,mpath,mapping):
    embedding_df = prep_embedding_df[["unique_index","gender","age_group"]]
    for col in mapping.keys():
        embedding_df[col] = prep_embedding_df[col].\
        apply(lambda x :sif_embeddings(str(x).split(" "), mpath))
    return embedding_df


