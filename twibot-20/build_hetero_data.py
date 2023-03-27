import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

import numpy as np
from datetime import datetime


def build_hetero_data() -> HeteroData:
    tmp_files_root = r"./preprocess/tmp-files"
    print(f"{datetime.now()}----Loading properties...")
    cat_props = torch.split(torch.load(rf"{tmp_files_root}/cat_props_tensor.pt"), 11826)[0]
    num_props = torch.split(torch.load(rf"{tmp_files_root}/num_props_tensor.pt"), 11826)[0]
    des = torch.split(torch.load(rf"{tmp_files_root}/des_tensor.pt"), 11826)[0]

    print(f"{datetime.now()}----Loading label...")
    label = torch.load(rf"{tmp_files_root}/label_tensor.pt")

    print(f"{datetime.now()}----Loading tweet...")
    tweet = torch.load(rf"{tmp_files_root}/tweet_tensor.pt")

    follow = torch.load(rf"{tmp_files_root}/follow_edge_index.pt")
    friend = torch.load(rf"{tmp_files_root}/friend_edge_index.pt")
    post = torch.load(rf"{tmp_files_root}/post_edge_index.pt")

    # train_index = torch.load(rf"{tmp_files_root}/train_index.pt").long()
    train_idx = np.array(torch.load(rf"{tmp_files_root}/train_index.pt"))
    train_index = torch.zeros(11826)
    train_index[train_idx] = 1
    train_index = train_index.bool()

    # val_index = torch.load(rf"{tmp_files_root}/val_index.pt").long()
    val_idx = np.array(torch.load(rf"{tmp_files_root}/val_index.pt"))
    val_index = torch.zeros(11826)
    val_index[val_idx] = 1
    val_index = val_index.bool()

    # test_index = torch.load(rf"{tmp_files_root}/test_index.pt").long()
    test_idx = np.array(torch.load(rf"{tmp_files_root}/test_index.pt"))
    test_index = torch.zeros(11826)
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
                'x': torch.concat([cat_props, num_props, des], dim=1),
                'y': label,
                'train_mask': train_index,
                'val_mask': val_index,
                'test_mask': test_index
            },
            'tweet': {
                'x': tweet,
                'test_mask': test_tweet_mask
            }
        },
        user__follow__user={'edge_index': follow},
        user__friend__user={'edge_index': friend},
        user__post__tweet={'edge_index': post}
    )
    undirected_transform = ToUndirected(merge=True)
    data = undirected_transform(data)
    return data

