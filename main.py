import processing as p
import embedding as e
import pca
import clustering as c
import autoencoder as n
import clusterindex as i
import pandas as pd
import ast
import os
from time import time
from args import args
import fpath as f


'''get table ready'''
print("get corpus...")
tartan_corpus, conv_ids, speaker_ids = p.load_corpus(f.corpus_dir)

if args.sampling == "valid":
    fname = os.path.join(f.prepdata_dir,'user_table_2020-4.json')
    df_utable = p.get_user_table(fname,conv_ids)
    valid_user_table = p.get_valid_table(df_utable)
    # default max sampling
    gp_para = ['gender', 'age_group']
    sampled_table = p.max_sampling(valid_user_table, gp_para)

if args.sampling == "all":
    valid_users = [[uid,cid] for cid in conv_ids \
               for uid in set(tartan_corpus.get_conversation(cid).get_speaker_ids())\
               if 'bot' not in uid]
    valid_user_table = pd.DataFrame(data=valid_users, columns = ['user_id','conversationId'])
    # no sampling
    sampled_table = valid_user_table

print("corpus ready")


# '''distribution plot'''
# gp_para = ['gender','age_group']
# p.get_distribution_chart(valid_user_table,gp_para)
# p.get_distribution_chart(sampled_table,gp_para)


'''utterance table'''
print("get utterance table...")
utt_table = p.get_utterance_table(tartan_corpus,sampled_table)
print(utt_table.head())
utt_table.to_csv(os.path.join(f.aggdata_dir,"utt_table_all.csv"))
print("utterance table ready")



'''pre embedding df'''
print("get pre-embedding table...")
mapping = e.get_mapping(utt_table)
prep_embedding_df = e.prompt_clasify(utt_table,mapping)
print(prep_embedding_df.iloc[:10,:])
prep_embedding_df.to_csv(os.path.join(f.aggdata_dir,"pre_embedding_table.csv"))
print("pre-embedding table ready")


'''embedding'''
mpath = "model/w2v_all.model"
indexes = prep_embedding_df.columns[:list(prep_embedding_df.columns).index("emotion")]

if args.reembed == "True":
    print("start re-embedding...")
    start = time()
    embedding_df = e.get_embedding_df(prep_embedding_df,indexes,mpath,mapping)
    X_train = pd.DataFrame.from_records(
        embedding_df.apply(
          lambda x: c.concat_ls(x,mapping), axis=1).values)
    print("embedding ready")
    end = time()
    print("time for embedding : {} s".format(end - start))
    embedding_df.to_csv(os.path.join(f.aggdata_dir,args.exportfp),index=False)


elif args.reembed == "False":
    print("start reading embedding table...")
    embedding_df = pd.read_csv(os.path.join(f.aggdata_dir,args.embedfp))
    for para in mapping.keys():
        embedding_df[para] = embedding_df[para].\
            apply(lambda x: [float(a) for a in x.replace("\n","").\
                  strip('][').\
                  split(" ") if a != ""])
    print("embedding ready")


'''
get pca df
'''
if args.pca == "True":
    print("get pca table...")
    pca_df = pca.get_pca_df(embedding_df,indexes,mapping.keys(),args.variance,args.fill_mean)
    print("pca table ready")

elif args.pca == "False":
    print("use the raw embedding df for clustering")
    pca_df = embedding_df

# print(pca_df.head())


'''
add conv meta info
'''
if args.meta == "True":
    print("adding meta data...")
    conv_meta_table = tartan_corpus.get_conversations_dataframe(). \
        reset_index(). \
        rename(columns={'id': 'conversationId'})
    conv_meta_table = p.get_meta(conv_meta_table)
    pca_df = pca_df.merge(conv_meta_table)
    print(pca_df.head())
    print(pca_df.columns)
    print("meta data added")

paras = pca_df.columns[list(pca_df.columns).index("emotion_0"):]

'''
scaler
'''
pca_df[paras] = p.scale(pca_df[paras])

'''clustering'''
if args.neural == "False":
    print("start clustering...")
    pca_result = c.multi_clustering(pca_df[paras],args.kmin,args.kmax,args.inter)
    pca_df[pca_result.columns] = pca_result
    pca_df.to_csv(os.path.join(f.pca_result,"pca_cluster_{}_{}_{}.csv".format(args.kmin,args.kmax,round(time()))),index=False)
    print("clustering successful! k ranges from {} to {}".format(args.kmin,args.kmax))
    # c.get_distribution_plot(pca_df,3,['gender'],['#e4e3e3','#84a9ac'])
    # c.get_distribution_plot(pca_df,3,['age_group'],['#f5f1da','#cedebd'])
    # c.get_distribution_plot(pca_df,3,['gender','age_group'],['#f5f1da','#cedebd','#e4e3e3','#84a9ac'])

else:
    print("start training nn model...")
    train_x,train_y = n.get_train_xy(pca_df,indexes,paras)
    autoencoder, encoder = n.train_model(train_x,args.mname,args.nntrain)
    nn_df = pd.DataFrame(n.predict_result(train_x,encoder))
    nn_df.columns = ["para_"+str(c) for c in nn_df.columns]
    print("nn result calculated!")
    print("start clustering...")
    pca_result = c.multi_clustering(nn_df, args.kmin, args.kmax)
    pca_df[pca_result.columns] = pca_result
    pca_df.to_csv(os.path.join(f.aggdata_dir,"pca_cluster_nn.csv"), index=False)
    print("clustering successful! k ranges from {} to {}".format(args.kmin,args.kmax))

'''compare dunn index among Ks'''
if args.dunn == "True":
    print("start calculating dunn indexes...")
    if "gender" in pca_df.columns:
        indexes = i.compare_dunn_index(pca_df.iloc[:,3:],k_max)
    else:
        indexes = i.compare_dunn_index(pca_df.iloc[:,1:],k_max)
    print(indexes)
    indexes.plot(x='dunn_index',y='num_of_cluster', marker='.')




