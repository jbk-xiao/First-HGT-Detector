import torch
from torch import nn
from torch_geometric.nn import HGTConv


class PropertyVector(nn.Module):
    def __init__(self, n_cat_prop=4, n_num_prop=5, des_size=768, embedding_dimension=128, fixed_size=4,
                 layer_norm_eps=1e-5, dropout=0.3):
        super(PropertyVector, self).__init__()

        self.n_cat_prop = n_cat_prop
        self.n_num_prop = n_num_prop
        self.des_size = des_size
        self.fixed_size = fixed_size

        self.cat_prop_module = nn.Sequential(
            nn.Linear(n_cat_prop, int(embedding_dimension / 4)),
            nn.LayerNorm(int(embedding_dimension / 4), eps=layer_norm_eps),
            nn.LeakyReLU()
        )
        self.num_prop_module = nn.Sequential(
            nn.Linear(n_num_prop, int(embedding_dimension / 4)),
            nn.LayerNorm(int(embedding_dimension / 4), eps=layer_norm_eps),
            nn.LeakyReLU()
        )
        # self.prop_module = nn.Sequential(
        #     nn.Linear(int(embedding_dimension / 2), int(embedding_dimension / 2)),
        #     nn.LeakyReLU()
        # )
        self.des_module = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LayerNorm(int(embedding_dimension / 4), eps=layer_norm_eps),
            nn.LeakyReLU()
        )
        self.consistency_module = nn.Sequential(
            nn.Linear(fixed_size * fixed_size, int(embedding_dimension / 4)),
            nn.LayerNorm(int(embedding_dimension / 4), eps=layer_norm_eps),
            nn.LeakyReLU()
        )
        self.out_layer = nn.Sequential(
            nn.LayerNorm(embedding_dimension, eps=layer_norm_eps),
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_tensor):
        cat_prop, num_prop, des, consistency = torch.split_with_sizes(
            user_tensor,
            [self.n_cat_prop, self.n_num_prop, self.des_size, self.fixed_size * self.fixed_size],
            dim=1
        )
        cat_prop_vec = self.dropout(self.cat_prop_module(cat_prop))
        num_prop_vec = self.dropout(self.num_prop_module(num_prop))
        prop_vec = torch.concat((cat_prop_vec, num_prop_vec), dim=1)
        # prop_vec = self.dropout(self.prop_module(prop_vec))
        des_vec = self.dropout(self.des_module(des))
        consistency_vec = self.dropout(self.consistency_module(consistency))
        profile_vec = torch.concat((prop_vec, des_vec, consistency_vec), dim=1)
        profile_vec = self.dropout(self.out_layer(profile_vec))
        return profile_vec


class SemanticConsistency(nn.Module):
    def __init__(self, tweet_size=768, num_heads=4, layer_norm_eps=1e-5, fixed_size=4, dropout=0.3):
        super(SemanticConsistency, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=tweet_size, num_heads=num_heads,
                                                          dropout=dropout, batch_first=True)
        self.update_text = nn.Sequential(
            nn.LayerNorm(tweet_size, eps=layer_norm_eps),
            nn.Linear(tweet_size, tweet_size),
            nn.LeakyReLU()
        )
        self.fixed_size = fixed_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_src):
        text, attention_weight = self.multi_head_attention(text_src, text_src, text_src)
        text = self.dropout(self.update_text(text_src + text))
        consistency = self.fixed_matrix(attention_weight).view(self.fixed_size * self.fixed_size)
        return text, consistency

    def fixed_matrix(self, attention_weight):
        w, h = attention_weight.shape
        if w <= self.fixed_size or h <= self.fixed_size:
            p_h = self.fixed_size - h
            p_w = self.fixed_size - w
            attention_weight = nn.functional.pad(attention_weight, (0, p_h, 0, p_w))
        else:
            p_h = self.fixed_size * ((h + self.fixed_size - 1) // self.fixed_size) - h
            p_w = self.fixed_size * ((w + self.fixed_size - 1) // self.fixed_size) - w
            attention_weight = nn.functional.pad(attention_weight, (0, p_h, 0, p_w))
            kernel_size = (
                ((w + self.fixed_size - 1) // self.fixed_size), ((h + self.fixed_size - 1) // self.fixed_size)
            )
            pool = nn.MaxPool2d(kernel_size, stride=kernel_size)
            attention_weight = torch.squeeze(pool(torch.unsqueeze(attention_weight, 0)))
        return attention_weight


class TweetVector(nn.Module):
    def __init__(self, tweet_size=768, embedding_dimension=128, dropout=0.3):
        super(TweetVector, self).__init__()
        self.tweet_module = nn.Sequential(
            nn.Linear(tweet_size, embedding_dimension),
            nn.LeakyReLU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tweet_tensor):
        tweet_vec = self.dropout(self.tweet_module(tweet_tensor))
        return tweet_vec


class HGTDetector(nn.Module):
    def __init__(self, n_cat_prop=4, n_num_prop=5, des_size=768, tweet_size=768, fixed_size=4, embedding_dimension=128,
                 dropout=0.3, num_heads=4, layer_norm_eps=1e-5):
        super(HGTDetector, self).__init__()

        meta_node = ["user", "tweet"]
        meta_edge = [
            ("user", "follow", "user"),
            ("user", "friend", "user"),
            ("user", "post", "tweet"),
            ("tweet", "rev_post", "user")
        ]
        self.fixed_size = fixed_size

        self.module_dict = nn.ModuleDict()
        self.module_dict["user"] = PropertyVector(
            n_cat_prop=n_cat_prop, n_num_prop=n_num_prop, des_size=des_size, embedding_dimension=embedding_dimension,
            fixed_size=fixed_size, layer_norm_eps=layer_norm_eps, dropout=dropout
        )
        self.module_dict["tweet"] = TweetVector(tweet_size, embedding_dimension, dropout)

        self.semantic_consistency = SemanticConsistency(tweet_size=tweet_size, num_heads=num_heads,
                                                        fixed_size=fixed_size, dropout=dropout,
                                                        layer_norm_eps=layer_norm_eps)

        self.HGT_layer1 = HGTConv(in_channels=embedding_dimension, out_channels=embedding_dimension,
                                  metadata=(meta_node, meta_edge), dropout=dropout)
        self.HGT_layer2 = HGTConv(in_channels=embedding_dimension, out_channels=embedding_dimension,
                                  metadata=(meta_node, meta_edge), dropout=dropout)

        self.classify_layer = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU(),
            nn.Linear(embedding_dimension, 2),
            nn.Softmax(dim=1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        user_text_dict = {}
        for user_idx in range(len(x_dict["user"])):
            user_text_dict[user_idx] = []
        for tweet_idx, user_idx in torch.transpose(edge_index_dict[("tweet", "rev_post", "user")], dim0=0, dim1=1).tolist():
            user_text_dict[user_idx].append(tweet_idx)
        for user_idx, tweet_idxs in user_text_dict.items():
            text, consistency = self.semantic_consistency(x_dict["tweet"][tweet_idxs])
            x_dict["tweet"][tweet_idxs] = text
            x_dict["user"][user_idx][-1 * self.fixed_size * self.fixed_size:] = consistency

        x_dict = {
            node_type: self.module_dict[node_type](x)
            for node_type, x in x_dict.items()
        }

        x_dict = self.HGT_layer1(x_dict, edge_index_dict)
        x_dict = self.HGT_layer2(x_dict, edge_index_dict)

        out = self.dropout(self.classify_layer(x_dict["user"]))

        return out
