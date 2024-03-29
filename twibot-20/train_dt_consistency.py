import copy
import random

import torch
from torch import nn
from torch_geometric.loader import NeighborLoader, HGTLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from similar_model import SimilarityModel
# from build_hetero_user_data import build_hetero_data
from build_hetero_user_data_v2 import build_hetero_data # v2 loads weighted tweets

tseed = torch.initial_seed()
tcseed = torch.cuda.initial_seed()
# torch.manual_seed(230523)
# torch.cuda.manual_seed(230523)
# torch.cuda.manual_seed_all(230523)

device = "cuda:0"
is_hgt_loader = False
use_random_mask = False
train_batch_size = 512
val_batch_size = 512
test_batch_size = 1
max_epoch = 20

model = SimilarityModel(n_cat_prop=4, n_num_prop=5, text_feature_dim=768, hidden_dim=512, hgt_layers=2, dropout=0)\
    .to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6, verbose=True)

print(f"{datetime.now()}----Loading data...")
data, des, user_tweets = build_hetero_data()

test_data = data.subgraph(
    {
        'user': ~(data['user'].train_mask + data['user'].val_mask)
    }
)

data = data.subgraph(
    {
        'user': ~data['user'].test_mask
    }
)
if is_hgt_loader:
    train_loader = HGTLoader(
        data,
        num_samples={key: [512] for key in data.node_types},
        shuffle=True,
        input_nodes=('user', data['user'].train_mask),
        batch_size=train_batch_size,
        num_workers=0)
    val_loader = HGTLoader(
        data,
        num_samples={key: [512] for key in data.node_types},
        shuffle=True,
        input_nodes=('user', data['user'].val_mask),
        batch_size=val_batch_size,
        num_workers=0)
    test_loader = HGTLoader(
        test_data,
        num_samples={key: [512] for key in data.node_types},
        shuffle=True,
        input_nodes=('user', test_data['user'].test_mask),
        batch_size=test_batch_size,
        num_workers=0)
else:
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [-1] for key in data.edge_types},
        shuffle=True,
        input_nodes=('user', data['user'].train_mask),
        batch_size=train_batch_size,
        num_workers=0)
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [-1] for key in data.edge_types},
        shuffle=True,
        input_nodes=('user', data['user'].val_mask),
        batch_size=val_batch_size,
        num_workers=0)
    test_loader = NeighborLoader(
        test_data,
        num_neighbors={key: [-1] for key in data.edge_types},
        shuffle=True,
        input_nodes=('user', test_data['user'].test_mask),
        batch_size=test_batch_size,
        num_workers=0)

print(f"{datetime.now()}----Data loaded.")


def forward_one_batch(batch, task):
    assert task in ['train', 'val', 'test']
    cur_batch_size = batch['user'].batch_size
    input_id = batch['user'].input_id
    if task == 'val':
        input_id += 8278
    if task == 'test':
        input_id += 10643
    cur_des = des[input_id]
    cur_tweets = user_tweets[input_id]
    if task == 'train' and use_random_mask:
        random_mask = random.randrange(0, 9)
        batch['user'].x[:, random_mask] = 0
        random_mask = random.choices(range(768), k=20)
        cur_des[:, random_mask] = 0
        random_mask = random.choices(range(768), k=20)
        cur_tweets[:, random_mask] = 0
    batch = batch.to(device)
    out = model(batch.x_dict, batch.edge_index_dict, cur_des.to(device), cur_tweets.to(device), cur_batch_size)
    return cur_batch_size, out


@torch.no_grad()
def init_params():
    batch = next(iter(train_loader))
    forward_one_batch(batch, 'train')


def train():
    model.train()

    total_examples = total_correct = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        cur_batch_size, out = forward_one_batch(batch, 'train')
        pred = out.argmax(dim=-1)
        total_correct += int((pred == batch['user'].y[0:cur_batch_size]).sum())
        loss = nn.functional.cross_entropy(out, batch['user'].y[0:cur_batch_size])
        loss.backward()
        optimizer.step()
        total_examples += cur_batch_size
        total_loss += float(loss) * cur_batch_size

    return (total_correct / total_examples), (total_loss / total_examples)


@torch.no_grad()
def val():
    model.eval()

    total_examples = total_correct = total_loss = 0
    for batch in tqdm(val_loader):
        cur_batch_size, out = forward_one_batch(batch, 'val')
        pred = out.argmax(dim=-1)
        total_correct += int((pred == batch['user'].y[0:cur_batch_size]).sum())
        loss = nn.functional.cross_entropy(out, batch['user'].y[0:cur_batch_size])
        total_examples += cur_batch_size
        total_loss += float(loss) * cur_batch_size

    return (total_correct / total_examples), (total_loss / total_examples)


@torch.no_grad()
def test(mode):
    assert mode in ['acc', 'loss']
    model.eval()

    label = []
    out = []
    for iteration, batch in enumerate(tqdm(test_loader)):
        cur_batch_size, pred = forward_one_batch(batch, 'test')
        pred = pred[0:cur_batch_size]
        pred = pred.argmax(dim=-1)
        out += pred.tolist()
        label += batch['user'].y[0:cur_batch_size].tolist()

    accuracy = accuracy_score(label, out)
    f1 = f1_score(label, out)
    precision = precision_score(label, out)
    recall = recall_score(label, out)

    print(f"Best {mode} Test, accuracy: {accuracy:.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")
    torch.save(model, rf'./saved_models/dt-acc{accuracy:.4f}.pickle')


init_params()
best_val_acc = 0.0
best_acc_loss = 9999.9
best_val_loss = 9999.9
best_loss_acc = 0.0
best_acc_epoch = 0
best_loss_epoch = 0
best_acc_model = ''
best_loss_model = ''
for epoch in range(1, max_epoch + 1):
    train_acc, loss = train()
    val_acc, val_loss = val()
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_acc_loss = val_loss
        best_acc_epoch = epoch
        best_acc_model = copy.deepcopy(model.state_dict())
    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_loss_acc = val_acc
        best_loss_epoch = epoch
        best_loss_model = copy.deepcopy(model.state_dict())
    scheduler.step(val_acc)
    print(f'Epoch: {epoch:03d}, Train_Acc: {train_acc:.4f}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Loss: {val_loss:.4f}')
print(f'Best val acc is: {best_val_acc:.4f}, in epoch: {best_acc_epoch:03d}, loss: {best_acc_loss:.4f}.')
print(f'Best val loss is: {best_val_loss:.4f}, in epoch: {best_loss_epoch:03d}, acc: {best_loss_acc:.4f}.')
model.load_state_dict(best_acc_model)
test('acc')
model.load_state_dict(best_loss_model)
test('loss')
print(tseed)
print(tcseed)

