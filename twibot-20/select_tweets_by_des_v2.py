import torch
from torch import nn

tmp_files_root = r"./preprocess/tmp-files"
des = torch.load(rf"{tmp_files_root}/des_tensor.pt")[0:11826]
tweet = torch.load(rf"{tmp_files_root}/tweet_tensor.pt")
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

tweet_similarity = torch.sum(user_tweets * des.unsqueeze(1), dim=2)  # shape: [11826, 200]

weights = (
        (tweet_similarity - tweet_similarity.min(dim=-1).values.unsqueeze(dim=-1))
        / (tweet_similarity.max(dim=-1).values - tweet_similarity.min(dim=-1).values).unsqueeze(dim=-1)
)
weights[weights.isnan()] = 0
weights = nn.Softmax(dim=-1)(- weights)

weighted_tweets = torch.sum(user_tweets * weights.unsqueeze(2), dim=1)  # shape: [11826, 768]

torch.save(weighted_tweets, rf'{tmp_files_root}/weighted_tweets_by_des.pt')
