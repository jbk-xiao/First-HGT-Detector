import torch
from torch import nn

tmp_files_root = r"./preprocess/tmp-files"
des = torch.load(rf"{tmp_files_root}/des_tensor.pt")[0:11826]
tweet = torch.load(rf"{tmp_files_root}/tweet_tensor.pt")
post = torch.load(rf"{tmp_files_root}/post_edge_index.pt")
user_tweets = []
for user_idx in range(11826):
    # user_text_dict[user_idx] = []
    user_tweets.append([])
for user_idx, tweet_idx in torch.transpose(post, dim0=0, dim1=1).tolist():
    user_tweets[user_idx].append(tweet[tweet_idx])
tweet_counts = []
for user_idx in range(11826):
    tweet_counts.append(len(user_tweets[user_idx]))
    if len(user_tweets[user_idx]) == 0:
        user_tweets[user_idx] = torch.zeros([1, 768])
    else:
        user_tweets[user_idx] = torch.stack(user_tweets[user_idx])

selected_tweets = []
for user_idx in range(11826):
    # similarity = torch.einsum('j, ij -> i', des[user_idx], user_tweets[user_idx]) # v0
    # similarity = torch.sum(user_tweets[user_idx] * des[user_idx].unsqueeze(0), dim=1) # v5
    similarity = torch.cosine_similarity(user_tweets[user_idx], des[user_idx].unsqueeze(0)) # v6
    if (similarity.max() - similarity.min()) != 0:
        weight = (similarity - similarity.min()) / (similarity.max() - similarity.min())
    else:
        weight = torch.zeros_like(similarity)
    weight = nn.Softmax(dim=-1)(- weight)
    weighted_tweets = torch.einsum('i, ij -> ij', weight, user_tweets[user_idx])
    # weighted_tweets = user_tweets[user_idx] * weight.unsqueeze(1)
    selected_tweets.append(weighted_tweets.sum(dim=0))
selected_tweets = torch.stack(selected_tweets)
torch.save(selected_tweets, rf'{tmp_files_root}/weighted_tweets_by_des_v0.pt')
