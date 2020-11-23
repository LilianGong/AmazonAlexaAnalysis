from argparse import ArgumentParser

parser = ArgumentParser(description ='user profiling')

parser.add_argument('--sampling', default = "all",
                    help = 'choose the sample size, valid or all',
                    type = str)
parser.add_argument('--reembed', default = "False",
                    help = 'choose whether re-calculate the embedding or not, True or False',
                    type = str)
parser.add_argument('--embedfp', default = "embedding_df.csv",
                    help = 'pre-embedded data fpath',
                    type = str)
parser.add_argument('--exportfp', default = "embedding_df.csv",
                    help = 'choose whether re-calculate the embedding or not, True or False',
                    type = str)
parser.add_argument('--pca', default = "True",
                    help = 'whether use pca result as input or not',
                    type = str)
parser.add_argument('--variance', default = 0.8,
                    help = 'how many percentage of variance should be kept in pca',
                    type = float)
parser.add_argument('--meta', default = "True",
                    help = 'whether includes meta data or not',
                    type = str)
parser.add_argument('--neural', default = "False",
                    help = 'whether use neuralnet or not',
                    type = str)
parser.add_argument('--nntrain', default = "False",
                    help = 'whether re-train the neuralnet model or not',
                    type = str)
parser.add_argument('--mname', default = "autoencoder",
                    help = 'nn model name',
                    type = str)
parser.add_argument('--kmin', default = 2,
                    help = 'min cluster k',
                    type = int)
parser.add_argument('--kmax', default = 10,
                    help = 'max cluster k',
                    type = int)

parser.add_argument('--inter', default = 1,
                    help = 'cluster interval' ,
                    type = int)

parser.add_argument('--dunn', default = "True",
                    help = 'calculate the dunn index',
                    type = str)

parser.add_argument('--fill_mean', default = "False",
                    help = 'fill na\'s as mean' ,
                    type = str)

args = parser.parse_args()