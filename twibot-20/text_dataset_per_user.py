import torch
from torch.utils.data import Dataset
import numpy as np

from datetime import datetime
import logging
from tqdm import tqdm

from config import ModelConfig

logger = logging.getLogger(__name__)
model_config = ModelConfig()


# 构建text_dataset，其中，每个item是一个用户的所有tweet连接起来
class TextDataset(Dataset):
    def __init__(self, action, remove_support=True):
        super().__init__()
        tmp_files_root = r"./preprocess/tmp-files"
        assert action in ['train', 'generate']
        if action == 'generate':
            remove_support = False
        self.action = action
        logger.info(f"TextDataset in {self.action} mode.")

        size_samples = 229850  # account count of Twibot-20
        logger.info(f"{datetime.now()}----Loading label...")
        label = torch.load(rf"{tmp_files_root}/label_tensor.pt")
        label_tensor = torch.ones(size_samples) * (-1)
        label_tensor[0:len(label)] = label
        label = label_tensor.long()
        del label_tensor

        train_idx = np.array(torch.load(rf"{tmp_files_root}/train_index.pt"))
        train_index = torch.zeros(size_samples)
        train_index[train_idx] = 1
        train_index = train_index.bool()

        val_idx = np.array(torch.load(rf"{tmp_files_root}/val_index.pt"))
        val_index = torch.zeros(size_samples)
        val_index[val_idx] = 1
        val_index = val_index.bool()

        test_idx = np.array(torch.load(rf"{tmp_files_root}/test_index.pt"))
        test_index = torch.zeros(size_samples)
        test_index[test_idx] = 1
        test_index = test_index.bool()

        logger.info(f"{datetime.now()}----Loading tweet...")
        print(f"{datetime.now()}----Loading tweet...")

        # word_vec = np.load(rf"{tmp_files_root}/vec.npy")  # 截取到content_bow_dim - 1大小，需要增加一行
        self.words_size = model_config.content_bow_dim - 1
        # 令所有大于conten_bow_dim - 1的值都等于 conten_bow_dim - 1
        tweets_per_user = np.load(rf"{tmp_files_root}/less_tweets.npy", allow_pickle=True)
        if remove_support:
            tweets_per_user = tweets_per_user[0:11826]

        tweet_sequences = []  # 不等长
        seq_lengths = []
        style_labels = []
        test_tweet_arr = []
        max_len = 0

        user_id = tweet_id = 0
        for tweet_per_user in tqdm(tweets_per_user, desc="Loading tweets..."):
            user_label = int(label[user_id])
            is_test = int(test_index[user_id])
            cat_tweet_per_user = []
            tweet_num_per_user = 0
            for each_tweet in tweet_per_user:
                for i in range(len(each_tweet)):
                    if each_tweet[i] > self.words_size:
                        each_tweet[i] = self.words_size
                cat_tweet_per_user += each_tweet
                tweet_num_per_user += 1
                if tweet_num_per_user > model_config.max_tweets_per_user:
                    break
            max_len = len(cat_tweet_per_user) if len(cat_tweet_per_user) > max_len else max_len
            if len(cat_tweet_per_user) > model_config.max_seq_len:
                tweet_sequences.append(cat_tweet_per_user[0:model_config.max_seq_len])
                seq_lengths.append(model_config.max_seq_len)
            else:
                tweet_sequences.append(cat_tweet_per_user if len(cat_tweet_per_user) > 0 else [self.words_size])
                seq_lengths.append(len(cat_tweet_per_user) if len(cat_tweet_per_user) > 0 else 1)
            style_label = [0, 0, 0]
            style_label[user_label + 1] = 1
            style_labels.append(style_label)
            test_tweet_arr.append(1 if is_test else 0)
            tweet_id += 1
            user_id += 1
        del tweets_per_user

        logger.info(f"{datetime.now()}----{tweet_id} tweets of {user_id} users loaded.")
        print(f"{datetime.now()}----{tweet_id} tweets of {user_id} users loaded.")
        test_tweet_mask = torch.tensor(test_tweet_arr).bool()
        self.tweets_length = tweet_id
        logger.info(f"tweets length: {self.tweets_length}, max length of each tweet: {max_len},"
                    f" config max_seq_len: {model_config.max_seq_len}.")
        print(f"tweets length: {self.tweets_length}, max length of each tweet: {max_len}, config max_seq_len:"
              f" {model_config.max_seq_len}.")
        self.max_len = max_len if max_len < model_config.max_seq_len else model_config.max_seq_len
        self.tweet_sequences = np.array(tweet_sequences, dtype=object)
        del tweet_sequences
        self.seq_lengths = torch.tensor(seq_lengths).long()
        self.style_labels = torch.tensor(style_labels).int()

        if action == 'train':
            self.tweets_length -= test_tweet_mask.sum()
            logger.info(f"train tweets length: {self.tweets_length}")
            self.tweet_sequences = self.tweet_sequences[~test_tweet_mask]
            self.seq_lengths = self.seq_lengths[~test_tweet_mask]
            self.style_labels = self.style_labels[~test_tweet_mask]

    def __getitem__(self, index):
        tweet_sequences = self.tweet_sequences[index]
        seq_lengths = self.seq_lengths[index]
        style_labels = self.style_labels[index]

        pad_tweet_sequences = np.ones(self.max_len) * self.words_size  # words_size = content_bow_dim - 1
        content_bow = torch.zeros(model_config.content_bow_dim)

        for word in tweet_sequences:
            if word < model_config.content_bow_dim:
                content_bow[word] += 1
        pad_tweet_sequences[0:int(seq_lengths)] = tweet_sequences
        tweet_sequences = torch.tensor(pad_tweet_sequences).int()

        return tweet_sequences, seq_lengths, style_labels, content_bow

    def get_iter(self) -> iter:
        for index in range(self.tweets_length):
            pad_tweet_sequences = np.ones(self.max_len) * self.words_size  # words_size = content_bow_dim - 1
            pad_tweet_sequences[0:int(self.seq_lengths[index])] = self.tweet_sequences[index]
            tweet_sequences = torch.tensor(pad_tweet_sequences).int()
            yield tweet_sequences, self.seq_lengths[index]

    def __len__(self):
        return self.tweets_length
