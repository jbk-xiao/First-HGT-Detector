#%%
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime
from transformers import pipeline
#%%
datasets_root = r"E:/social-bot-data/datasets/Twibot-20"
tmp_files_root = r"./tmp-files"
#%%
print(f"{datetime.now()}----Reading node2id.csv...")
node2id_list = pd.read_csv(rf"{datasets_root}/node2id.csv", dtype={"node_id": str, "num_id": int})
# tweets: 1-33488192, users: 33488193-33713010
#%%
node_file = "mini-nodes-for-test.json"
#%% md
### 较为快速地生成推文向量，内存占用较高
#%%
print(f"{datetime.now()}----Reading node.json...")
node_df = pd.read_json(rf"{datasets_root}/{node_file}", encoding="utf-8")
tweet_df = (node_df[node_df.id.str.len() > 0])[node_df.id.str.contains("^t")]
tweet_df = pd.merge(tweet_df, node2id_list, left_on="id", right_on="node_id", how="inner")
tweet_df.sort_values("num_id", ascending=True, inplace=True)
#%%
tweet_feature_extract = pipeline('feature-extraction', model='roberta-base', tokenizer='roberta-base', device=0, padding=True, truncation=True, max_length=50, add_special_tokens=True)
#%%
print(f"{datetime.now()}----Generate tweet tensors...")
tweet_list = []

def get_tweet_tensor(text):
    if text is None:
        tweet_list.append(torch.zeros(768))
    else:
        word_tensors = torch.tensor(tweet_feature_extract(text))
        each_tweet_tensor = torch.zeros(768)
        for word_tensor in word_tensors[0]:
            each_tweet_tensor += word_tensor
        tweet_list.append(each_tweet_tensor)

tqdm.pandas(desc="get tweet tensors")
tweet_df["text"].progress_apply(get_tweet_tensor)
tweet_tensor = torch.stack(tweet_list, 0)
torch.save(tweet_tensor, rf"{tmp_files_root}/tweet_tensor.pt")

