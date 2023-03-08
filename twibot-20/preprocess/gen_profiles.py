#%%
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime
from transformers import pipeline
#%%
datasets_root = r"E:/social-bot-data/datasets/Twibot-20"
tmp_files_root = r"./tmp-files"
#%%
node2id_list = pd.read_csv(rf"{datasets_root}/node2id.csv", dtype={"node_id": str, "num_id": int})
# tweets: 1-33488192, users: 33488193-33713010
# node2id = {}
# for row in tqdm(node2id_list.iterrows(), desc="Generate node2id dict."):
#     node2id[row[1]["node_id"]] = row[1]["num_id"]
#%%
node_file = "mini-nodes-for-test.json"
#%% md
### 读取用户node文件并排序
#%%
print(f"{datetime.now()}----Reading node.json...")
node_df = pd.read_json(rf"{datasets_root}/{node_file}", encoding="utf-8")
user_df = (node_df[node_df.id.str.len() > 0])[node_df.id.str.contains("^u")]
user_df = pd.merge(user_df, node2id_list, left_on="id", right_on="node_id", how="inner")
user_df.sort_values("num_id", ascending=True, inplace=True)
#%% md
'''
### 生成用户类别型属性的表示，n_cat_prop = 4
四种类别型属性：
- is_verified
- is_protected
- is_default_profile_image
- has_tweets
#%%
'''
print(f"{datetime.now()}----Generate category properties...")
is_verified_list = []
is_protected_list = []
is_default_profile_image_list = []
has_tweets_list = []

tqdm.pandas(desc="is verified list")
user_df["verified"].progress_apply(lambda x: is_verified_list.append(1) if x == "True " else is_verified_list.append(0))
tqdm.pandas(desc="is protected list")
user_df["protected"].progress_apply(lambda x: is_protected_list.append(1) if x == "True " else is_protected_list.append(0))
default_profile_image_url = "http://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png "
tqdm.pandas(desc="is default image url list")
user_df["profile_image_url"].progress_apply(lambda x: is_default_profile_image_list.append(1) if x == default_profile_image_url
else is_default_profile_image_list.append(0))
def check_has_tweets_list(public_metrics):
    # public_metrics = pd.DataFrame(public_metrics)
    if public_metrics is not None and isinstance(public_metrics, dict):
        if public_metrics["tweet_count"] is not None and public_metrics["tweet_count"] != 0:
            has_tweets_list.append(1)
        else:
            has_tweets_list.append(0)
    else:
        has_tweets_list.append(0)
tqdm.pandas(desc="check has tweets list")
user_df["public_metrics"].progress_apply(check_has_tweets_list)

cat_props = np.transpose([is_verified_list, is_protected_list, is_default_profile_image_list, has_tweets_list])
cat_props_tensor = torch.tensor(cat_props, dtype=torch.float)
torch.save(cat_props_tensor, rf"{tmp_files_root}/cat_props_tensor.pt")
'''
#%% md
### 生成用户数量型属性的表示，n_num_prop = 5
- followers_count
- following_count
- listed_count (status)
- active_days
- screen_name_length
#%%
'''
print(f"{datetime.now()}----Generate numerical properties...")
followers_count_list = []
following_count_list = []
listed_count_list = []
active_days_list = []
screen_name_length_list = []

def get_public_metrics_num(public_metrics):
    if public_metrics is not None and isinstance(public_metrics, dict):
        if public_metrics["followers_count"] is not None:
            followers_count_list.append(int(public_metrics["followers_count"]))
        else:
            followers_count_list.append(0)

        if public_metrics["following_count"] is not None:
            following_count_list.append(int(public_metrics["following_count"]))
        else:
            following_count_list.append(0)

        if public_metrics["listed_count"] is not None:
            listed_count_list.append(int(public_metrics["listed_count"]))
        else:
            listed_count_list.append(0)

    else:
        followers_count_list.append(0)
        following_count_list.append(0)
        listed_count_list.append(0)
tqdm.pandas(desc="get numerical properties from public_metrics")
user_df["public_metrics"].progress_apply(get_public_metrics_num)

# created_at = pd.to_datetime(user_df["created_at"], unit='s')
init_date = datetime.strptime("Tue Sep 1 00:00:00 +0000 2020 ", "%a %b %d %X %z %Y ")
tqdm.pandas(desc="get active days from created_at")
user_df["created_at"].progress_apply(lambda x: active_days_list.append(int((init_date - x).days)))

tqdm.pandas(desc="get the length of username")
user_df["username"].progress_apply(lambda x: screen_name_length_list.append(len(x)) if x is not None
else screen_name_length_list.append(0))

def fill_and_z_score(num_prop_list):
    num_prop_df = pd.DataFrame(num_prop_list).fillna(int(0))
    num_prop = (num_prop_df - num_prop_df.mean()) / num_prop_df.std()
    num_prop_tensor = torch.tensor(np.array(num_prop), dtype=torch.float32)
    return num_prop_tensor

followers_count_tensor = fill_and_z_score(followers_count_list)
following_count_tensor = fill_and_z_score(following_count_list)
listed_count_tensor = fill_and_z_score(listed_count_list)
active_days_tensor = fill_and_z_score(active_days_list)
screen_name_length_tensor = fill_and_z_score(screen_name_length_list)

num_props_tensor = torch.cat([followers_count_tensor, following_count_tensor, listed_count_tensor, active_days_tensor,
                              screen_name_length_tensor], dim=1)
torch.save(num_props_tensor, rf"{tmp_files_root}/num_props_tensor.pt")
#%% md
### 生成用户描述的表示
#%%
des_feature_extract = pipeline('feature-extraction', model='roberta-base', tokenizer='roberta-base', device=0, padding=True, truncation=True, max_length=50, add_special_tokens=True)
#%%
print(f"{datetime.now()}----Generate description...")
des_list = []

def get_des_tensor(description):
    if description is None:
        des_list.append(torch.zeros(768))
    else:
        word_tensors = torch.tensor(des_feature_extract(description))
        each_des_tensor = torch.zeros(768)
        for word_tensor in word_tensors[0]:
            each_des_tensor += word_tensor
        des_list.append(each_des_tensor)

tqdm.pandas(desc="get description tensors")
user_df["description"].progress_apply(get_des_tensor)
des_tensor = torch.stack(des_list, 0)
torch.save(des_tensor, rf"{tmp_files_root}/des_tensor.pt")

