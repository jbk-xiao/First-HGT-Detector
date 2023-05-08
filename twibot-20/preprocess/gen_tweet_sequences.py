import json
import ijson
import numpy as np
import pandas
from nltk import tokenize
from tqdm import tqdm
import jieba

from nltk.corpus import stopwords
from sklearn.feature_extraction import _stop_words as stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from gensim.models import word2vec

datasets_root = r"E:/social-bot-data/datasets/Twibot-20"
tmp_files_root = r"./tmp-files"


def get_tweet_corpus():
    nltk_stopwords = set(stopwords.words('english'))
    sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS
    all_stopwords = set()
    # The '|' operator on sets in python acts as a union operator
    all_stopwords |= spacy_stopwords
    all_stopwords |= nltk_stopwords
    all_stopwords |= sklearn_stopwords
    all_stopwords.add('https')
    all_stopwords.add('http')

    def is_allowed(word: str) -> bool:
        if word.find('@') != -1 or word.find('#') != -1:
            return False
        if len(word) > 15 or len(word) == 0:
            return False
        # if word is not unicode? if word is not start with alpha or number?
        if not ('a' <= word[0] <= 'z' or 'A' <= word[0] <= 'Z' or '0' <= word[0] <= '9'):
            return False
        if word in all_stopwords:
            return False
        return True
    fb = open(rf'{tmp_files_root}/corpus.txt', 'w')
    with open(rf'{datasets_root}/node.json') as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            if item['id'].find('t') == -1:
                continue
            if item['text'] is None:
                continue
            line = item['text']
            word_list = [word for word in jieba.cut(line) if is_allowed(word)]
            line = ' '.join(word_list)
            fb.write(line + '\n')
    fb.close()


def get_word2vec_model():
    sentences = word2vec.Text8Corpus(rf'{tmp_files_root}/corpus.txt')
    print('Training word2vec model...')
    model = word2vec.Word2Vec(sentences, vector_size=128, workers=4, min_count=10)
    vectors = model.wv.vectors
    key_to_index = model.wv.key_to_index
    index_to_key = model.wv.index_to_key
    print(f'vectors.shape: {vectors.shape}')
    print(f'len(key_to_index): {len(key_to_index)}, len(index_to_key): {len(index_to_key)}')
    np.save(rf'{tmp_files_root}/less_vec.npy', vectors)
    json.dump(key_to_index, open(rf'{tmp_files_root}/less_key_to_index.json', 'w', encoding='utf-8'))
    json.dump(index_to_key, open(rf'{tmp_files_root}/less_index_to_key.json', 'w', encoding='utf-8'))


def get_tweet_sequences():
    edge = pandas.read_csv(rf'{datasets_root}/edge.csv')
    author_idx = {}
    for index, item in tqdm(edge.iterrows(), ncols=0, desc='Reading post edges...'):
        if item['relation'] != 'post':
            continue
        author_idx[item['target_id']] = item['source_id']
    print(len(edge))
    key_to_index = json.load(open(rf'{tmp_files_root}/less_key_to_index.json'))
    user_idx = pandas.read_csv(rf'{datasets_root}/node2id.csv')
    user_idx = user_idx[user_idx['node_id'].str.contains('^u')]['node_id'].tolist()
    tweets_index = {}
    for user in user_idx:
        tweets_index[user] = []
    max_len1 = max_len2 = 0
    with open(rf'{datasets_root}/node.json') as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            if item['id'].find('t') == -1:
                continue
            if item['text'] is None:
                continue
            words = tokenize.word_tokenize(item['text'])
            max_len1 = len(words) if len(words) > max_len1 else max_len1
            tweet = []
            for word in words:
                if word in key_to_index:
                    tweet.append(key_to_index[word])
                else:
                    tweet.append(len(key_to_index))
            tweets_index[author_idx[item['id']]].append(tweet)
            max_len2 = len(tweet) if len(tweet) > max_len2 else max_len2
    print(f'max_len1: {max_len1}, max_len2: {max_len2}.')  # max_len=754
    tweets = [tweets_index[item] for item in user_idx]
    tweets = np.array(tweets, dtype=object)
    np.save(rf'{tmp_files_root}/less_tweets.npy', tweets)

