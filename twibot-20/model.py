import torch
from torch import nn
from torch_geometric.nn import HGTConv


class PropertyVector(nn.Module):
    def __init__(self, n_cat_prop=4, n_num_prop=5, des_size=768, embedding_dimension=128, dropout=0.3):
        super(PropertyVector, self).__init__()

        self.n_cat_prop = n_cat_prop
        self.n_num_prop = n_num_prop
        self.des_size = des_size

        self.cat_prop_module = nn.Sequential(
            nn.Linear(n_cat_prop, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.num_prop_module = nn.Sequential(
            nn.Linear(n_num_prop, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.prop_module = nn.Sequential(
            nn.Linear(int(embedding_dimension / 2), int(embedding_dimension / 2)),
            nn.LeakyReLU()
        )
        self.des_module = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 2)),
            nn.LeakyReLU()
        )
        self.out_layer = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

    def forward(self, user_tensor):
        cat_prop, num_prop, des = torch.split_with_sizes(user_tensor, [self.n_cat_prop, self.n_num_prop, self.des_size], dim=1)
        cat_prop_vec = self.cat_prop_module(cat_prop)
        num_prop_vec = self.num_prop_module(num_prop)
        des_vec = self.des_module(des)
        prop_vec = torch.concat((cat_prop_vec, num_prop_vec, des_vec), dim=1)
        prop_vec = self.out_layer(prop_vec)
        return prop_vec


class TweetVector(nn.Module):
    def __init__(self, tweet_size=768, embedding_dimension=128, dropout=0.3):
        super(TweetVector, self).__init__()
        self.tweet_module = nn.Sequential(
            nn.Linear(tweet_size, embedding_dimension),
            nn.LeakyReLU()
        )

    def forward(self, tweet_tensor):
        tweet_vec = self.tweet_module(tweet_tensor)
        return tweet_vec


class HGTDetector(nn.Module):
    def __init__(self, n_cat_prop=4, n_num_prop=5, des_size=768, tweet_size=768, embedding_dimension=128, dropout=0.3):
        super(HGTDetector, self).__init__()

        meta_node = ["user", "tweet"]
        meta_edge = [("user", "follow", "user"), ("user", "friend", "user"), ("user", "post", "tweet")]

        self.module_dict = nn.ModuleDict()
        self.module_dict["user"] = PropertyVector(n_cat_prop, n_num_prop, des_size, embedding_dimension, dropout)
        self.module_dict["tweet"] = TweetVector(tweet_size, embedding_dimension, dropout)

        self.HGT_layer1 = HGTConv(in_channels=embedding_dimension, out_channels=embedding_dimension, metadata=(meta_node, meta_edge))

        self.classify_layer = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
            nn.Linear(embedding_dimension, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.module_dict[node_type](x)
            for node_type, x in x_dict.items()
        }

        x_dict = self.HGT_layer1(x_dict, edge_index_dict)

        out = self.classify_layer(x_dict["user"])

        return out

