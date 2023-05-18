import json

import gensim.utils
import ijson
import numpy as np
import pandas
from nltk import tokenize
from tqdm import tqdm

from nltk.corpus import stopwords
from sklearn.feature_extraction import _stop_words as stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from gensim.models import word2vec

datasets_root = r"E:/social-bot-data/datasets/Twibot-20"
tmp_files_root = r"./tmp-files"


def get_tweet_corpus():
    fb = open(rf'{tmp_files_root}/corpus.txt', 'w')
    with open(rf'{datasets_root}/node.json') as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            if item['id'].find('t') == -1:
                continue
            if item['text'] is None:
                continue
            line = item['text']
            fb.write(line + '\n')
    fb.close()


def trim_rule(word: str, count: int, min_count: int) -> int:

    nltk_stopwords = set(stopwords.words('english'))
    sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS
    all_stopwords = set()
    # The '|' operator on sets in python acts as a union operator
    all_stopwords |= spacy_stopwords
    all_stopwords |= nltk_stopwords
    all_stopwords |= sklearn_stopwords
    all_stopwords.add('https')
    all_stopwords.add('http')

    if count < min_count:
        return gensim.utils.RULE_DISCARD
    if word.find('@') != -1 or word.find('#') != -1:
        return gensim.utils.RULE_DISCARD
    if len(word) > 15 or len(word) == 0:
        return gensim.utils.RULE_DISCARD
    # if word is not unicode? if word is not start with alpha or number?
    if not ('a' <= word[0] <= 'z' or 'A' <= word[0] <= 'Z' or '0' <= word[0] <= '9'):
        return gensim.utils.RULE_DISCARD
    if word in all_stopwords:
        return gensim.utils.RULE_DISCARD
    return gensim.utils.RULE_KEEP


def get_word2vec_model():
    sentences = word2vec.Text8Corpus(rf'{tmp_files_root}/corpus.txt')
    print('Training word2vec model...')
    model = word2vec.Word2Vec(sentences, vector_size=128, workers=4, trim_rule=trim_rule, min_count=10)
    vectors = model.wv.vectors
    key_to_index = model.wv.key_to_index
    index_to_key = model.wv.index_to_key
    print(f'vectors.shape: {vectors.shape}')
    print(f'len(key_to_index): {len(key_to_index)}, len(index_to_key): {len(index_to_key)}')
    np.save(rf'{tmp_files_root}/less_vec.npy', vectors)
    json.dump(key_to_index, open(rf'{tmp_files_root}/less_key_to_index.json', 'w', encoding='utf-8'))
    json.dump(index_to_key, open(rf'{tmp_files_root}/less_index_to_key.json', 'w', encoding='utf-8'))


def get_des_sequences():
    key_to_index = json.load(open(rf'{tmp_files_root}/less_key_to_index.json'))
    user_idx = pandas.read_csv(rf'{datasets_root}/node2id.csv')
    user_idx = user_idx[user_idx['node_id'].str.contains('^u')]['node_id'].tolist()
    des_index = {}
    for user in user_idx:
        des_index[user] = []
    max_len1 = max_len2 = 0
    with open(rf'{datasets_root}/node.json') as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, total=229850):
            if item['id'].find('u') == -1:
                break
            if item['description'] is None:
                continue
            words = tokenize.word_tokenize(item['description'])
            max_len1 = len(words) if len(words) > max_len1 else max_len1
            des = []
            for word in words:
                if word in key_to_index:
                    des.append(key_to_index[word])
                else:
                    des.append(len(key_to_index))
            if len(des) != 0:  # gen tweet sequences has same error.
                des_index[item['id']].append(des)
            max_len2 = len(des) if len(des) > max_len2 else max_len2
    print(f'max_len1: {max_len1}, max_len2: {max_len2}.')  # max_len=77
    des_seq = [des_index[item] for item in user_idx]
    des_seq = np.array(des_seq, dtype=object)
    np.save(rf'{tmp_files_root}/des_seq.npy', des_seq)

