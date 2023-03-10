import torch
from torch import nn
from torch_geometric.nn import HGTConv


class PropertyVector(nn.Module):
    def __int__(self, n_cat_prop_size=4, n_num_prop=5, des_size=768, embedding_dimension=128, dropout=0.3):
        super(PropertyVector, self).__init__()

    def forward(self, cat_prop, num_prop, des):
        return


class TweetVector(nn.Module):
    def __int__(self, tweet_size=768, embedding_dimension=128, dropout=0.3):
        super(TweetVector, self).__int__()

    def forward(self, text):
        return


class HGTDetector(nn.Module):
    def __init__(self, n_cat_prop_size=4, n_num_prop=5, des_size=768, tweet_size=768, embedding_dimension=128, dropout=0.3):
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

        self.propertyVector = PropertyVector()
        self.tweetVector = TweetVector()
        self.HGT_lay1 = HGTConv(in_channels=embedding_dimension, out_channels=embedding_dimension, metadata=(meta_node, meta_edge))

    def forward(self, node_type, cat_prop, num_prop, des, text, edge_index, edge_type):
        if node_type == 0:
            node = self.propertyVector(cat_prop, num_prop, des)
            node_dict = {"user": node}
        else:
            node = self.tweetVector(text)
            node_dict = {"tweet": node}

        node = self.HGT_lay1(node_dict, self.edge_index_dict)

        return node

