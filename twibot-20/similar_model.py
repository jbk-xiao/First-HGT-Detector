import torch
from torch import nn
from torch_geometric.nn import HGTConv


class GraphEmbedding(nn.Module):
    def __init__(self, hidden_channels, hidden_dim, hgt_layers=2, dropout=0):
        super(GraphEmbedding, self).__init__()
        meta_data = (["user"], [("user", "follow", "user"), ("user", "friend", "user")])
        self.convs = nn.ModuleList(
            [HGTConv(in_channels=hidden_channels, out_channels=hidden_channels, metadata=meta_data, dropout=dropout)
             for _ in range(hgt_layers)]
        )
        self.graph_emb_projection = nn.Sequential(nn.Linear(hidden_channels, hidden_dim), nn.LeakyReLU())
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        graph_emb = self.dropout(self.graph_emb_projection(x_dict['user']))
        return graph_emb


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


class DesTweetConsistency(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout=0):
        super(DesTweetConsistency, self).__init__()

        # 可学习参数矩阵
        self.weight_matrix = nn.Parameter(torch.randn((200, 200), requires_grad=True))

        # 相似度计算层
        self.similarity_layer = nn.Linear(feature_dim, feature_dim)

        # 加权求和层
        self.weighted_sum_layer = nn.Linear(feature_dim, feature_dim)

        # 输出值映射
        self.dropout = nn.Dropout(p=dropout)
        self.des_projection = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LeakyReLU())
        self.weights_projection = nn.Sequential(nn.Linear(200, hidden_dim), nn.LeakyReLU())
        self.weighted_tweets_projection = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LeakyReLU())

    def forward(self, des, tweets):
        # 计算des和tweets中每条tweet的相似度
        des_similarity = self.similarity_layer(des)  # shape: [batch_size, feature_dim]
        tweet_similarity = self.similarity_layer(tweets)  # shape: [batch_size, 200, feature_dim]
        tweet_similarity = torch.sum(tweet_similarity * des_similarity.unsqueeze(1), dim=2)  # shape: [batch_size, 200]

        # 计算tweets中每一条tweet的权重
        weights = (
                (tweet_similarity - tweet_similarity.min(dim=-1).values.unsqueeze(dim=-1))
                / (tweet_similarity.max(dim=-1).values - tweet_similarity.min(dim=-1).values).unsqueeze(dim=-1)
        )
        weights[weights.isnan()] = 0
        weights = nn.Softmax(dim=-1)(- weights)
        l_weights = torch.matmul(weights, self.weight_matrix)  # shape: [batch_size, 200]

        # 加权求和
        weighted_tweets = self.weighted_sum_layer(tweets)  # shape: [batch_size, 200, feature_dim]
        weighted_tweets = torch.sum(weighted_tweets * l_weights.unsqueeze(2), dim=1)  # shape: [batch_size, feature_dim]

        # 输出值映射层
        des = self.dropout(self.des_projection(des))
        weights = self.dropout(self.weights_projection(weights))
        weighted_tweets = self.dropout(self.weighted_tweets_projection(weighted_tweets))

        return des, weights, weighted_tweets


class BotClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0):
        super(BotClassifier, self).__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU())
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, 2), nn.LeakyReLU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature):
        feature = self.dropout(self.hidden_layer(feature))
        return self.dropout(self.output_layer(feature))


class SimilarityModel(nn.Module):
    def __init__(self, n_cat_prop, n_num_prop, text_feature_dim, hidden_dim, hgt_layers, dropout=0):
        super(SimilarityModel, self).__init__()
        self.graph_embedding = GraphEmbedding(
            hidden_channels=(n_cat_prop + n_num_prop), hidden_dim=hidden_dim, hgt_layers=hgt_layers, dropout=dropout
        )
        self.property_embedding = PropertyEmbedding(
            input_dim=(n_cat_prop + n_num_prop), hidden_dim=hidden_dim, dropout=dropout
        )
        # self.des_tweets_embedding = DesTweetConsistency(
        #     feature_dim=text_feature_dim, hidden_dim=hidden_dim, dropout=dropout
        # )
        self.bot_classifier = BotClassifier(input_dim=hidden_dim * 2 + text_feature_dim * 2, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x_dict, edge_index_dict, des, tweets):
        batch_size = des.shape[0]
        user_props = x_dict['user'][:batch_size]
        graph_emb = self.graph_embedding(x_dict, edge_index_dict)[:batch_size]
        prop_emb = self.property_embedding(user_props)
        # des_emb, consistency_emb, weighted_tweets_emb = self.des_tweets_embedding(des, tweets)
        user_feature = torch.cat([graph_emb, prop_emb, des, tweets], dim=1)
        return nn.Softmax(dim=-1)(self.bot_classifier(user_feature))

