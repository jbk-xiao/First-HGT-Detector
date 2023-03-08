#%%
import pandas as pd
import torch
import tqdm
import ijson
from transformers import pipeline
#%%
datasets_root = r"E:\social-bot-data\datasets\Twibot-20"
tmp_files_root = r"E:\social-bot-data\code\First-HGT-Detector\twibot-20\preprocess\tmp-files"
#%%
node2id_list = pd.read_csv(rf"{datasets_root}\node2id.csv", dtype={"node_id": str, "num_id": int})
# tweets: 1-33488192, users: 33488193-33713010
node2id = {}
for row in node2id_list.iterrows():
    node2id[row[1]["node_id"]] = row[1]["num_id"]
#%% md
### 利用node文件按顺序生成所有推文的向量表示
#%%
tweet_feature_extract = pipeline('feature-extraction', model='roberta-base', tokenizer='roberta-base', device=0,
                                 padding=True, truncation=True, max_length=50, add_special_tokens=True)
#%%
tweet_tensors_dicts = []
with open(rf"{datasets_root}\mini-nodes-for-test.json", 'r', encoding="utf-8") as f:
    for record in tqdm(ijson.items(f, "item"), desc="Reading node.json with each item."):
        if record.get("text"):
            word_tensors = torch.tensor(tweet_feature_extract(record.get("text")))
            each_tweet_tensor = torch.zeros(768)
            for each_word_tensor in word_tensors[0]:
                each_tweet_tensor += each_word_tensor
            tweet_tensors_dicts.append({"node_id": record.get("id"), "tweet_tensor": each_tweet_tensor})

tweet_tensors_df = pd.DataFrame(tweet_tensors_dicts)
tweet_tensors_df = pd.merge(tweet_tensors_df, node2id_list, on="node_id", how="inner")
tweet_tensors_df.sort_values(by="num_id", inplace=True, ascending=True)
#%%
tweet_tensors_df.to_pickle(rf"{tmp_files_root}\tweet_tensors_df.pkl")
tweet_tensors = torch.stack(tweet_tensors_df["tweet_tensor"].tolist())
torch.save(tweet_tensors, rf"{tmp_files_root}\tweet_tensors.pt")

