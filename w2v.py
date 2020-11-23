from gensim.models import Word2Vec
from convokit import Corpus, Speaker, Utterance
from tqdm import tqdm

tartan_corpus = Corpus(filename="../tartan_corpus")

# gather all user utterances as training data
# 360,345 utterances
utterances = []
utt_id = tartan_corpus.get_utterance_ids()
for _id in tqdm(utt_id):
    utt = tartan_corpus.get_utterance(_id)
    if 'user' in utt._id:
        utterances.append(utt.text.split(" "))

# w2v model training
model = Word2Vec(utterances, min_count=1)
model.save("model/w2v_all.model")