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
    des = des[0:11826]  # shape: [11826, word_emb_dim]
    weighted_tweet = torch.load(rf"{tmp_files_root}/weighted_tweets_by_des.pt")
    user_profiles = torch.concat([cat_props, num_props], dim=1)
    size_samples = user_profiles.size(dim=0)  # int

    print(f"{datetime.now()}----Loading label...")
    label = torch.load(rf"{tmp_files_root}/label_tensor.pt")  # shape: [11826]
    label_tensor = torch.ones(size_samples) * (-1)  # shape: [229850]
    label_tensor[0:len(label)] = label
    label = label_tensor.long()

    follow = torch.load(rf"{tmp_files_root}/follow_edge_index.pt")
    friend = torch.load(rf"{tmp_files_root}/friend_edge_index.pt")

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
    return data, des, weighted_tweet

