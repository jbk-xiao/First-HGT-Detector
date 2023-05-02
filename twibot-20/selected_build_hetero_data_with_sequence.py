import torch
from numpy import ndarray
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

import numpy as np
import json
from datetime import datetime

from tqdm import tqdm
from config import ModelConfig


def build_hetero_data(remove_profiles=True, fixed_size=4) -> tuple[HeteroData, ndarray[list], int, Tensor, dict]:
    tmp_files_root = r"./preprocess/tmp-files"
    print(f"{datetime.now()}----Loading properties...")
    model_config = ModelConfig()
    vocab_size = model_config.vocab_size
    content_bow_dim = model_config.content_bow_dim
    # cat_props = torch.split(torch.load(rf"{tmp_files_root}/cat_props_tensor.pt"), 11826)[0]
    # num_props = torch.split(torch.load(rf"{tmp_files_root}/num_props_tensor.pt"), 11826)[0]
    # des = torch.split(torch.load(rf"{tmp_files_root}/des_tensor.pt"), 11826)[0]
    cat_props = torch.load(rf"{tmp_files_root}/cat_props_tensor.pt")
    num_props = torch.load(rf"{tmp_files_root}/num_props_tensor.pt")
    des = torch.load(rf"{tmp_files_root}/des_tensor.pt")
    user_profiles = torch.concat([cat_props, num_props, des], dim=1)
    size_samples = user_profiles.size(dim=0)  # int
    if remove_profiles:
        user_profiles = torch.zeros_like(user_profiles)

    print(f"{datetime.now()}----Loading label...")
    label = torch.load(rf"{tmp_files_root}/label_tensor.pt")
    label_tensor = torch.ones(size_samples) * (-1)
    label_tensor[0:len(label)] = label
    label = label_tensor.long()
    del label_tensor

    follow = torch.load(rf"{tmp_files_root}/follow_edge_index.pt")
    friend = torch.load(rf"{tmp_files_root}/friend_edge_index.pt")
    # post = torch.load(rf"{tmp_files_root}/post_edge_index.pt")

    # train_index = torch.load(rf"{tmp_files_root}/train_index.pt").long()
    train_idx = np.array(torch.load(rf"{tmp_files_root}/train_index.pt"))
    train_index = torch.zeros(size_samples)
    train_index[train_idx] = 1
    train_index = train_index.bool()

    # val_index = torch.load(rf"{tmp_files_root}/val_index.pt").long()
    val_idx = np.array(torch.load(rf"{tmp_files_root}/val_index.pt"))
    val_index = torch.zeros(size_samples)
    val_index[val_idx] = 1
    val_index = val_index.bool()

    # test_index = torch.load(rf"{tmp_files_root}/test_index.pt").long()
    test_idx = np.array(torch.load(rf"{tmp_files_root}/test_index.pt"))
    test_index = torch.zeros(size_samples)
    test_index[test_idx] = 1
    test_index = test_index.bool()

    print(f"{datetime.now()}----Loading tweet...")

    word_vec = np.load(rf"{tmp_files_root}/vec.npy")  # 截取到content_bow_dim - 1大小，需要增加一行
    words_size = content_bow_dim - 1
    tweets_per_user = np.load(rf"{tmp_files_root}/tweets.npy", allow_pickle=True) # 令所有大于conten_bow_dim - 1的值都等于 conten_bow_dim - 1
    key_to_index = json.load(open(rf"{tmp_files_root}/key_to_index.json"))  # 是否有用

    tweet_sequences = []  # 不等长
    seq_lengths = []
    style_labels = []
    test_tweet_arr = []
    max_len = 0
    post_arr = []  # [[user_id0, tweet_id0], [user_id0, tweet_id1], ......], shape = (tweet_num, 2)

    user_id = tweet_id = 0
    for tweet_per_user in tqdm(tweets_per_user, desc="Loading tweets..."):
        user_label = int(label[user_id])
        is_test = int(test_idx[user_id])
        for each_tweet in tweet_per_user:
            max_len = len(each_tweet) if len(each_tweet) > max_len else max_len
            for i, _ in enumerate(each_tweet):
                if each_tweet[i] > words_size:
                    each_tweet[i] = words_size
            tweet_sequences.append(each_tweet)
            seq_lengths.append(len(each_tweet))
            style_label = [0, 0, 0]
            style_label[user_label + 1] = 1
            style_labels.append(style_label)
            test_tweet_arr.append(1 if is_test else 0)
            post_arr.append([user_id, tweet_id])
            tweet_id += 1
        user_id += 1

    # pad_tweet_sequences = np.ones((tweet_id, max_len)) * words_size
    # 以最后一个word_vec填充不等长list，内存会炸，取一个batch后再填充
    # for i in tqdm(range(tweet_id), desc="Padding tweet_sequences..."):
    #     pad_tweet_sequences[i][0:seq_lengths[i]] = tweet_sequences[i]
    # del tweet_sequences
    print(f"{datetime.now()}----{tweet_id} tweets of {user_id} users loaded.")

    word_vec = torch.tensor(word_vec)[0:words_size]
    blank_vec = torch.zeros((1, word_vec.shape[-1]))
    word_vec = torch.cat((word_vec, blank_vec), dim=0)
    tweet = torch.zeros([tweet_id, 128])
    tweet_index = torch.arange(0, tweet_id).int()
    tweet_sequences = np.array(tweet_sequences, dtype=object)
    seq_lengths = torch.tensor(seq_lengths).long()
    style_labels = torch.tensor(style_labels).int()
    test_tweet_mask = torch.tensor(test_tweet_arr).bool()
    post = torch.tensor(np.transpose(post_arr))


    print(f"{datetime.now()}----Building data...")
    data = HeteroData(
        {
            'user': {
                'x': user_profiles,
                'y': label,
                'train_mask': train_index,
                'val_mask': val_index,
                'test_mask': test_index
            },
            'tweet': {
                'x': tweet,
                'tweet_index': tweet_index,
                'seq_length': seq_lengths,
                'style_label': style_labels,
                'test_mask': test_tweet_mask
            }
        },
        user__follow__user={'edge_index': follow},
        user__friend__user={'edge_index': friend},
        user__post__tweet={'edge_index': post}
    )
    undirected_transform = ToUndirected(merge=True)
    data = undirected_transform(data)
    return data, tweet_sequences, max_len, word_vec, key_to_index

