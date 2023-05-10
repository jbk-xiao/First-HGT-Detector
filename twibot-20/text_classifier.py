import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

device = 'cuda:0'
tweet_emb_file = 'vae-2-5368.065876.pickle-tweet_embs.pt'
lr = 1e-4
epoch = 2


class TextClassifier(nn.Module):
    def __init__(self, embedding_dimension=128, hidden_dimension=256, output_dimension=2, dropout=0.3):
        super(TextClassifier, self).__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(embedding_dimension, hidden_dimension), nn.LeakyReLU())
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dimension, output_dimension),
            nn.LeakyReLU(),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_embedding):
        hidden_state = self.dropout(self.hidden_layer(text_embedding))
        out = self.dropout(self.output_layer(hidden_state))
        return out


class TextDataset(Dataset):
    def __init__(self, name):
        self.name = name
        if self.name == 'train':
            self.len = 8278
        else:
            self.len = 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.name == 'train':
            return index
        elif self.name == 'val':
            return torch.arange(8278, 10643)
        elif self.name == 'test':
            return torch.arange(10643, 11826)


train_dataset = TextDataset('train')
val_dataset = TextDataset('val')
test_dataset = TextDataset('test')

train_loader = DataLoader(train_dataset, batch_size=512)
val_loader = DataLoader(val_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)

tweet_emb = torch.load(rf'preprocess/tmp-files/{tweet_emb_file}').to(device)
label = torch.load(rf'preprocess/tmp-files/label_tensor.pt')[0:11826].to(device)
text_classifier = TextClassifier(embedding_dimension=128, hidden_dimension=256, output_dimension=2, dropout=0).to(device)
params = text_classifier.parameters()
optimizer = torch.optim.AdamW(params, lr=lr)

text_classifier(tweet_emb[next(iter(train_loader))])
best_acc = 0
best_epoch = 0
best_model_dict = {}
for i in range(epoch):
    total_examples = total_correct = total_loss = 0
    for train_index in tqdm(train_loader):
        optimizer.zero_grad()
        text_classifier.train()
        out = text_classifier(tweet_emb[train_index])
        loss = nn.functional.cross_entropy(out, label[train_index])
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=-1)
        total_correct += int((pred == label[train_index]).sum())
        total_examples += out.shape[0]
        total_loss += float(loss) * out.shape[0]
    print(rf"Epoch {i+1:03d}, train_acc: {total_correct / total_examples:.4f}, train_loss: {total_loss / total_examples:.4f}.")

    total_examples = total_correct = total_loss = 0
    for val_index in tqdm(val_loader):
        text_classifier.eval()
        val_index = val_index.squeeze()
        out = text_classifier(tweet_emb[val_index])
        loss = nn.functional.cross_entropy(out, label[val_index])
        pred = out.argmax(dim=-1)
        total_correct += int((pred == label[val_index]).sum())
        total_examples += out.shape[0]
        total_loss += float(loss) * out.shape[0]
    print(rf"Epoch {i+1:03d}, val_acc: {total_correct / total_examples:.4f}, val_loss: {total_loss / total_examples:.4f}.")
    if total_correct / total_examples > best_acc:
        best_acc = total_correct / total_examples
        best_epoch = i+1
        best_model_dict = copy.deepcopy(text_classifier.state_dict())

print(f"Best val acc is: {best_acc:.4f}, in epoch: {best_epoch:03d}.")
text_classifier.load_state_dict(best_model_dict)
total_examples = total_correct = total_loss = 0
pred = []
test_label = []
for test_index in tqdm(test_loader):
    text_classifier.eval()
    test_index = test_index.squeeze()
    out = text_classifier(tweet_emb[test_index])
    pred += out.argmax(dim=-1).tolist()
    test_label += label[test_index].tolist()
accuracy = accuracy_score(test_label, pred)
f1 = f1_score(test_label, pred)
precision = precision_score(test_label, pred)
recall = recall_score(test_label, pred)
print(f"Test: accuracy: {accuracy:.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")
torch.save(text_classifier, rf'./saved_models/text_only-acc{accuracy:.4f}.pickle')
