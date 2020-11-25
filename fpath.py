import os

root_dir = os.path.abspath('.')

corpus_dir = "../tartan_corpus"

prepdata_dir = os.path.join(root_dir, 'prepdata')

aggdata_dir = os.path.join(root_dir, 'aggdata')
nntrain = os.path.join(aggdata_dir, 'nntrain')

result_dir = os.path.join(root_dir, 'result')
nn_result = os.path.join(result_dir,"neuralnet")
som_result = os.path.join(result_dir,"som")
pca_result = os.path.join(result_dir,"pca")

model_dir = os.path.join(root_dir, 'model')
nnmodel = os.path.join(model_dir, 'neuralnet')
sommodel = os.path.join(model_dir, 'som')

anaysis = os.path.join(root_dir,"analysis")