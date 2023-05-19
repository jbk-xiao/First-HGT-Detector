import torch
from numpy import ndarray
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

import numpy as np
from datetime import datetime


def build_hetero_data() -> tuple[HeteroData, ndarray]:
    tmp_files_root = r"./preprocess/tmp-files"
    print(f"{datetime.now()}----Loading properties...")
    # cat_props = torch.split(torch.load(rf"{tmp_files_root}/cat_props_tensor.pt"), 11826)[0]
    # num_props = torch.split(torch.load(rf"{tmp_files_root}/num_props_tensor.pt"), 11826)[0]
    # des = torch.split(torch.load(rf"{tmp_files_root}/des_tensor.pt"), 11826)[0]
    cat_props = torch.load(rf"{tmp_files_root}/cat_props_tensor.pt")
    num_props = torch.load(rf"{tmp_files_root}/num_props_tensor.pt")
    des = torch.load(rf"{tmp_files_root}/des_tensor.pt")
    user_profiles = torch.concat([cat_props, num_props, des], dim=1)
    size_samples = user_profiles.size(dim=0)  # int

    print(f"{datetime.now()}----Loading label...")
    label = torch.load(rf"{tmp_files_root}/label_tensor.pt")
    label_tensor = torch.zeros(size_samples) * (-1)
    label_tensor[0:len(label)] = label
    label = label_tensor

    print(f"{datetime.now()}----Loading tweet...")
    tweet = torch.load(rf"{tmp_files_root}/tweet_tensor.pt")

    follow = torch.load(rf"{tmp_files_root}/follow_edge_index.pt")
    friend = torch.load(rf"{tmp_files_root}/friend_edge_index.pt")
    post = torch.load(rf"{tmp_files_root}/post_edge_index.pt")

    user_tweets = []
    for user_idx in range(11826):
        # user_text_dict[user_idx] = []
        user_tweets.append([])
    for user_idx, tweet_idx in torch.transpose(post, dim0=0, dim1=1).tolist():
        user_tweets[user_idx].append(tweet[tweet_idx])
    for user_idx in range(11826):
        if len(user_tweets[user_idx]) == 0:
            user_tweets[user_idx] = torch.zeros([1, 768])
        else:
            user_tweets[user_idx] = torch.stack(user_tweets[user_idx])
    user_tweets = np.array(user_tweets, dtype=object)

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

    test_tweet_idx = torch.load(rf"{tmp_files_root}/test_tweets_id.pt")
    test_tweet_mask = torch.zeros(tweet.size(dim=0))
    test_tweet_mask[test_tweet_idx] = 1
    test_tweet_mask = test_tweet_mask.bool()

    print(f"{datetime.now()}----Building data...")
    data = HeteroData(
        {
            'user': {
                'x': user_profiles,
                'y': label,
                'train_mask': train_index,
                'val_mask': val_index,
                'test_mask': test_index
            }
        },
        user__follow__user={'edge_index': follow},
        user__friend__user={'edge_index': friend}
    )
    undirected_transform = ToUndirected(merge=True)
    data = undirected_transform(data)
    return data, user_tweets

