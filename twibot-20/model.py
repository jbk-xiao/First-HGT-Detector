import torch
from torch import nn
from torch_geometric.nn import HGTConv


class PropertyVector(nn.Module):
    def __int__(self, n_cat_prop=4, n_num_prop=5, des_size=768, embedding_dimension=128, dropout=0.3):
        super(PropertyVector, self).__init__()
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
        self.out_layer = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

    def forward(self, cat_prop, num_prop, des):
        cat_prop_vec = self.cat_prop_module(cat_prop)
        num_prop_vec = self.num_prop_module(num_prop)
        prop_vec = self.prop_module(torch.concat((cat_prop_vec, num_prop_vec), dim=1))
        prop_vec = self.out_layer(prop_vec)
        return prop_vec


class TweetVector(nn.Module):
    def __int__(self, tweet_size=768, embedding_dimension=128, dropout=0.3):
        super(TweetVector, self).__int__()
        self.tweet_module = nn.Sequential(
            nn.Linear(tweet_size, embedding_dimension),
            nn.LeakyReLU()
        )

    def forward(self, text):
        tweet_vec = self.tweet_module(text)
        return tweet_vec


class HGTDetector(nn.Module):
    def __init__(self, n_cat_prop=4, n_num_prop=5, des_size=768, tweet_size=768, embedding_dimension=128, dropout=0.3):
        super(HGTDetector, self).__init__()

        edge_index = torch.load("edge_index.pt", map_location="cuda")
        edge_type = torch.load("edge_type.pt", map_location="cuda").long()
        meta_node = ["user", "tweet"]
        meta_edge = [("user", "follow", "user"), ("user", "friend", "user"), ("user", "post", "tweet")]
        follow_edge_index = edge_index[:, edge_type == 0]
        friend_edge_index = edge_index[:, edge_type == 1]
        post_edge_index = edge_index[:, edge_type == 2]
        self.edge_index_dict = {("user", "follow", "user"): follow_edge_index,
                                ("user", "friend", "user"): friend_edge_index,
                                ("user", "post", "tweet"): post_edge_index}

        self.propertyVector = PropertyVector(n_cat_prop, n_num_prop, des_size, embedding_dimension, dropout)
        self.tweetVector = TweetVector(tweet_size, embedding_dimension, dropout)

        self.HGT_layer1 = HGTConv(in_channels=embedding_dimension, out_channels=embedding_dimension, metadata=(meta_node, meta_edge))

        self.classify_layer = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
            nn.Linear(embedding_dimension, 2)
        )

    def forward(self, node_type, cat_prop, num_prop, des, text):
        if node_type == 0:
            node = self.propertyVector(cat_prop, num_prop, des)
            node_dict = {"user": node}
        else:
            node = self.tweetVector(text)
            node_dict = {"tweet": node}

        node = self.HGT_lay1(node_dict, self.edge_index_dict)

        node = self.classify_layer(node)

        return node

