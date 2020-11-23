from convokit import Corpus, Speaker, Utterance
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import re

'''load corpus'''
def load_corpus(fname):
    tartan_corpus = Corpus(filename=fname)
    conv_ids = tartan_corpus.get_conversation_ids()
    speaker_ids = tartan_corpus.get_speaker_ids()
    return tartan_corpus, conv_ids, speaker_ids

'''load user table'''
def get_user_table(fname,conv_ids):
    f = open(fname)
    utable = json.load(f)
    new_utable = {}
    for i in range(len(utable)):
        temp_table = {'user_id':utable[i][0]['user_id']}
        temp_table.update(utable[i][0]['map_attributes'])
        new_utable[i] = temp_table
    new_utable[100]
    df_utable = pd.DataFrame.from_dict(new_utable,orient = 'index')
    df_utable['filter'] = [True if x in conv_ids else False for x in df_utable['conversationId'] ]
    #df_utable.head()
    return df_utable


'''get valid user with gender & age infomation'''
def get_valid_table(df_utable):
    valid_user_table = df_utable[df_utable['filter']==True].\
                dropna(subset = ['gender','age_group'],axis = 0)
    valid_user_table['counter'] = 1
    print("altogether there are {} valid users with age & gender info".
          format((len(pd.unique(valid_user_table['user_id'])))))
    return valid_user_table


'''distribution '''
def get_distribution_chart(table, gp_para):
    num_of_charts = len(gp_para) + 1
    nrows = 2
    ncols = np.ceil(num_of_charts / nrows)

    fig, axs = plt.subplots(figsize=(10 * nrows, 5 * ncols))
    plt.axis('off')
    for i in range(num_of_charts - 1):
        fig.add_subplot(nrows, ncols, i + 1)
        table.groupby(gp_para[i]). \
            sum()['counter'].plot.bar(rot=1, color='#cedebd')

    fig.add_subplot(nrows, ncols, num_of_charts)
    table.groupby(gp_para). \
        sum()['counter'].plot.bar(rot=1, color='#cedebd')
    plt.show()


'''adjustment of sampling - method 1'''
def max_sampling(valid_user_table,gp_para):
    aimed_num = np.max(valid_user_table.\
                     groupby(gp_para).\
                     sum()['counter'])

    sampled_table = pd.DataFrame(columns = valid_user_table.columns)
    for name, group in valid_user_table.groupby(gp_para):
        sampled_gp = group.sample(aimed_num,random_state=8,replace=True)
        sampled_table = pd.concat([sampled_table,sampled_gp])
    return sampled_table


'''get utterance table - bot prompt first, user utt second'''
def get_utterance_table(corpus, sampled_table):
    u = []
    b = []
    uid = []
    cid = []
    ids = []
    id = 1
    for u_id, conv_id in sampled_table[['user_id','conversationId']].values:
        conv = corpus.get_conversation(conv_id)
        utt_id = conv.get_utterance_ids()
        i = 0
        while i < len(utt_id)-1:
            if i ==0:
                i+=1
            else:
                speaker_1 = re.search(r"_.*",utt_id[i]).\
                group(0).strip(r"_[0123456789]*_")
                speaker_2 = re.search(r"_.*",utt_id[i+1]).\
                group(0).strip(r"_[0123456789]*_")
                if speaker_1 != speaker_2:
                    if speaker_1=="bot":
                        b.append(conv.get_utterance(utt_id[i]).text)
                        u.append(conv.get_utterance(utt_id[i+1]).text)
                        uid.append(u_id)
                        cid.append(conv_id)
                        ids.append(id)
                        i+=2
                    else:
#                         print("{}:{}\n {}:{}".format\
#                               (speaker_1,conv.get_utterance(utt_id[i]).text,
#                                 speaker_2,conv.get_utterance(utt_id[i+1]).text))
                        i += 2
                else:
#                     print("consequtive utterance found")
#                     print("{}:{}\n {}:{}".format\
#                           (speaker_1,conv.get_utterance(utt_id[i]).text,
#                             speaker_2,conv.get_utterance(utt_id[i+1]).text))
                    i += 1
        id +=1
    utt_table = pd.DataFrame({"unique_index":ids,"conversationId":cid,"user_id":uid,"bot":b,"user":u})
    utt_table = utt_table.merge(sampled_table,how='inner').drop_duplicates()
    return utt_table


