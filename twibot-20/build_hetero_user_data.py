import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

import numpy as np
from datetime import datetime


def build_hetero_data() -> tuple[HeteroData, Tensor, Tensor]:
    tmp_files_root = r"./preprocess/tmp-files"
    print(f"{datetime.now()}----Loading properties...")
    # cat_props = torch.split(torch.load(rf"{tmp_files_root}/cat_props_tensor.pt"), 11826)[0]
    # num_props = torch.split(torch.load(rf"{tmp_files_root}/num_props_tensor.pt"), 11826)[0]
    # des = torch.split(torch.load(rf"{tmp_files_root}/des_tensor.pt"), 11826)[0]
    cat_props = torch.load(rf"{tmp_files_root}/cat_props_tensor.pt")  # shape: [229850, 4]
    num_props = torch.load(rf"{tmp_files_root}/num_props_tensor.pt")  # shape: [229850, 5]
    des = torch.load(rf"{tmp_files_root}/des_tensor.pt")  # shape: [229850, word_emb_dim]
    user_profiles = torch.concat([cat_props, num_props], dim=1)
    size_samples = user_profiles.size(dim=0)  # int

    print(f"{datetime.now()}----Loading label...")
    label = torch.load(rf"{tmp_files_root}/label_tensor.pt")  # shape: [11826]
    label_tensor = torch.ones(size_samples) * (-1)  # shape: [229850]
    label_tensor[0:len(label)] = label
    label = label_tensor.long()

    print(f"{datetime.now()}----Loading tweet...")
    tweet = torch.load(rf"{tmp_files_root}/tweet_tensor.pt")

    follow = torch.load(rf"{tmp_files_root}/follow_edge_index.pt")
    friend = torch.load(rf"{tmp_files_root}/friend_edge_index.pt")
    post = torch.load(rf"{tmp_files_root}/post_edge_index.pt")

    user_tweets = [[]] * 11826
    tweet_counts = [0] * 11826
    for user_idx, tweet_idx in torch.transpose(post, dim0=0, dim1=1).tolist():
        user_tweets[user_idx].append(tweet[tweet_idx])
        tweet_counts[user_idx] += 1
    del tweet, user_idx, tweet_idx
    for user_idx in range(11826):
        user_tweets[user_idx] = user_tweets[user_idx][0:min(tweet_counts[user_idx], 200)]
        for _ in range(min(tweet_counts[user_idx], 200), 200):
            user_tweets[user_idx].append(torch.zeros([768]))
        user_tweets[user_idx] = torch.stack(user_tweets[user_idx])
    user_tweets = torch.stack(user_tweets)  # shape: [11826, 200, 768]

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
    return data, des, user_tweets

