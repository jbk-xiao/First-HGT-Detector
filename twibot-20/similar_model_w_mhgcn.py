import math

import torch
from torch import nn
from torch_geometric.nn import HGTConv


# class GraphEmbedding(nn.Module):
#     def __init__(self, hidden_channels, hidden_dim, hgt_layers=2, dropout=0):
#         super(GraphEmbedding, self).__init__()
#         meta_data = (["user"], [("user", "follow", "user"), ("user", "friend", "user")])
#         self.convs = nn.ModuleList(
#             [HGTConv(in_channels=hidden_channels, out_channels=hidden_channels, metadata=meta_data, dropout=dropout)
#              for _ in range(hgt_layers)]
#         )
#         self.graph_emb_projection = nn.Sequential(nn.Linear(hidden_channels, hidden_dim), nn.LeakyReLU())
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x_dict, edge_index_dict):
#         for conv in self.convs:
#             x_dict = conv(x_dict, edge_index_dict)
#
#         graph_emb = self.dropout(self.graph_emb_projection(x_dict['user']))
#         return graph_emb


class PropertyEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0):
        super(PropertyEmbedding, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(p=dropout))

        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, user_props):
        x = user_props
        for layer in self.layers:
            x = layer(x)
        prop_emb = self.output_layer(x)
        return prop_emb


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        std_v = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std_v, std_v)
        if self.bias is not None:
            self.bias.data.uniform_(-std_v, std_v)

    def forward(self, x_feature, mh_adj_matrix):
        try:
            x_feature = x_feature.float()
        except:
            pass
        support = torch.mm(x_feature, self.weight)
        output = torch.spmm(mh_adj_matrix, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MHGCN(nn.Module):
    def __init__(self, in_feature, out_feature, n_gc_layers=2, n_relation=2, dropout=0):
        super(MHGCN, self).__init__()
        self.gc_layers = nn.ModuleList()
        for i in range(n_gc_layers):
            if i == 0:
                # self.gc_layers.append(nn.Sequential(GraphConvolution(in_feature, out_feature), nn.ReLU()))
                self.gc_layers.append(GraphConvolution(in_feature, out_feature))
            else:
                # self.gc_layers.append(nn.Sequential(GraphConvolution(out_feature, out_feature), nn.ReLU()))
                self.gc_layers.append(GraphConvolution(out_feature, out_feature))
        self.dropout = nn.Dropout(p=dropout)
        self.relation_weight = nn.Parameter(torch.FloatTensor(n_relation, 1), requires_grad=True)
        torch.nn.init.uniform_(self.relation_weight, a=0, b=0.1)

    def forward(self, x_feature, all_adj_matrix):
        mh_adj_matrix = torch.matmul(all_adj_matrix, self.relation_weight)
        mh_adj_matrix = mh_adj_matrix.squeeze(dim=2)
        mh_adj_matrix = mh_adj_matrix + mh_adj_matrix.transpose(0, 1)
        gc_output_list = []
        gc_output = x_feature
        for gc_layer in self.gc_layers:
            gc_output = gc_layer(gc_output, mh_adj_matrix)
            gc_output_list.append(gc_output)
        gc_output = torch.stack(gc_output_list, dim=0)
        gc_output = gc_output.mean(dim=0)
        return gc_output


class BotClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0):
        super(BotClassifier, self).__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU())
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, 2), nn.LeakyReLU(), nn.Softmax(dim=-1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature):
        feature = self.dropout(self.hidden_layer(feature))
        return self.dropout(self.output_layer(feature))


class SimilarityModel(nn.Module):
    def __init__(self, n_cat_prop, n_num_prop, text_feature_dim, hidden_dim, hgt_layers, dropout=0):
        super(SimilarityModel, self).__init__()
        # self.graph_embedding = GraphEmbedding(
        #     hidden_channels=(n_cat_prop + n_num_prop), hidden_dim=hidden_dim, hgt_layers=hgt_layers, dropout=dropout
        # )
        self.graph_embedding = MHGCN(
            in_feature=(n_cat_prop + n_num_prop), out_feature=hidden_dim, n_gc_layers=hgt_layers, dropout=dropout
        )
        self.property_embedding = PropertyEmbedding(
            input_dim=(n_cat_prop + n_num_prop), hidden_dim=hidden_dim, dropout=dropout
        )
        # self.des_tweets_embedding = DesTweetConsistency(
        #     feature_dim=text_feature_dim, hidden_dim=hidden_dim, dropout=dropout
        # )
        self.bot_classifier = BotClassifier(input_dim=hidden_dim * 2 + text_feature_dim * 1, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x_feature, adj_matrix, des, tweets, batch_size):
        user_props = x_feature[:batch_size]
        graph_emb = self.graph_embedding(x_feature, adj_matrix)[:batch_size]
        prop_emb = self.property_embedding(user_props)
        # des_emb, consistency_emb, weighted_tweets_emb = self.des_tweets_embedding(des, tweets)
        # user_feature = torch.cat([graph_emb, prop_emb, des, tweets], dim=1)
        user_feature = torch.cat([graph_emb, prop_emb, tweets], dim=1)
        return self.bot_classifier(user_feature)

