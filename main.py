import processing as p
import embedding as e
import pca
import clustering as c
import clusterindex as i
import pandas as pd
import ast

'''get table ready'''
print("get corpus...")
fname = "../tartan_corpus"
tartan_corpus, conv_ids, speaker_ids = p.load_corpus(fname)
fname = 'prepdata/user_table_2020-4.json'
df_utable = p.get_user_table(fname,conv_ids)
valid_user_table = p.get_valid_table(df_utable)
print("corpus ready")


# '''distribution plot'''
# gp_para = ['gender','age_group']
# p.get_distribution_chart(valid_user_table,gp_para)


'''sample table'''
gp_para = ['gender','age_group']
sampled_table = p.max_sampling(valid_user_table,gp_para)
# print(sampled_table.head())
# print(sampled_table.shape)
# p.get_distribution_chart(sampled_table,gp_para)


'''utterance table'''
print("get utterance table...")
utt_table = p.get_utterance_table(tartan_corpus,sampled_table)
print(utt_table.shape)
utt_table.head()
print("utterance table ready")


'''pre embedding df'''
print("get pre-embedding table...")
mapping = e.get_mapping(utt_table)
prep_embedding_df = e.prompt_clasify(utt_table,mapping)
print(prep_embedding_df.iloc[:10,:])
print("pre-embedding table ready")


'''embedding'''
mpath = "model/w2v_all.model"
try:
    embedding_df = pd.read_csv("aggdata/embedding_df.csv")
    for para in mapping.keys():
        embedding_df[para] = embedding_df[para].\
            apply(lambda x: [float(a) for a in x.replace("\n","").\
                  strip('][').\
                  split(" ") if a != ""])
except:
    print("embedding_df not found, start embedding...")
    embedding_df = e.get_embedding_df(prep_embedding_df,mpath,mapping)
    X_train = pd.DataFrame.from_records(
        embedding_df.apply(
          lambda x: c.concat_ls(x,mapping), axis=1).values)
    print("embedding ready")
    X_train.to_csv('aggdata/embedding_df.csv',index=False)


'''get pca df'''
print("get pca table...")
pca_df = pca.get_pca_df(embedding_df,mapping,0.8)
print("pca table ready")

'''clustering'''
print("start clustering...")
k_max = 10
pca_df = c.multi_clustering(pca_df.iloc[:,3:],k_max)
pca_df.to_csv("aggdata/pca_cluster.csv",index=False)
print("clustering successful! k ranges from 2 to {}".format(k_max))


'''compare dunn index among Ks'''
print("start calculating dunn indexes...")
indexes = i.compare_dunn_index(pca_df.iloc[:,3:],k_max)
print(indexes)
indexes.plot(x='dunn_index',y='num_of_cluster', marker='.')


# c.get_distribution_plot(pca_df,3,['gender'],['#e4e3e3','#84a9ac'])
# c.get_distribution_plot(pca_df,3,['age_group'],['#f5f1da','#cedebd'])
# c.get_distribution_plot(pca_df,3,['gender','age_group'],['#f5f1da','#cedebd','#e4e3e3','#84a9ac'])

