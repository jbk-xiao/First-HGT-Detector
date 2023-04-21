import copy
import random

import torch
from torch import nn
from torch_geometric.loader import NeighborLoader, HGTLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model import HGTDetector
from build_hetero_data import build_hetero_data

device = "cuda:0"
is_hgt_loader = False
fixed_size = 4
use_random_mask = False
remove_profiles = True

model = HGTDetector(n_cat_prop=4, n_num_prop=5, des_size=768, tweet_size=768,
                    embedding_dimension=128, dropout=0.3).to(device)


print(f"{datetime.now()}----Loading data...")
data = build_hetero_data(remove_profiles=remove_profiles, fixed_size=fixed_size)

test_data = data.subgraph(
    {
        'user': data['user'].test_mask,
        'tweet': data['tweet'].test_mask
    }
)

data = data.subgraph(
    {
        'user': ~data['user'].test_mask,
        'tweet': ~data['tweet'].test_mask
    }
)
if is_hgt_loader:
    train_loader = HGTLoader(
        data,
        num_samples={key: [512] for key in data.node_types},
        shuffle=True,
        input_nodes=('user', data['user'].train_mask),
        batch_size=512,
        num_workers=0)
    val_loader = HGTLoader(
        data,
        num_samples={key: [512] for key in data.node_types},
        shuffle=True,
        input_nodes=('user', data['user'].val_mask),
        batch_size=128,
        num_workers=0)
    test_loader = HGTLoader(
        test_data,
        num_samples={key: [512] for key in test_data.node_types},
        shuffle=True,
        input_nodes=('user', test_data['user'].test_mask),
        batch_size=1,
        num_workers=0)
else:
    train_loader = NeighborLoader(
        data,
        num_neighbors={key: [-1] for key in data.edge_types},
        shuffle=True,
        input_nodes=('user', data['user'].train_mask),
        batch_size=512,
        num_workers=0)
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [-1] for key in data.edge_types},
        shuffle=True,
        input_nodes=('user', data['user'].val_mask),
        batch_size=128,
        num_workers=0)
    test_loader = NeighborLoader(
        test_data,
        num_neighbors={key: [-1] for key in test_data.edge_types},
        shuffle=True,
        input_nodes=('user', test_data['user'].test_mask),
        batch_size=1,
        num_workers=0)


print(f"{datetime.now()}----Data loaded.")


@torch.no_grad()
def init_params():
    batch = next(iter(train_loader))
    batch = batch.to(device)
    model(batch.x_dict, batch.edge_index_dict)


def train(lr):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=lr)
    total_examples = total_correct = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        if use_random_mask:
            random_mask = random.randrange(0, 9)
            batch['user'].x[:, random_mask] = 0
            random_mask = random.choices(range(768), k=20)
            batch['user'].x[:, [x + 9 for x in random_mask]] = 0
            random_mask = random.choices(range(768), k=20)
            batch['tweet'].x[:, random_mask] = 0

        # batch_size = batch['user'].batch_size
        train_mask = batch['user'].train_mask
        out = model(batch.x_dict, batch.edge_index_dict)[train_mask]
        # print(f"out[train_mask]: {out}")
        # print(f"out[train_mask].argmax(-1): {out.argmax(dim=-1)}")
        # print(f"batch['user'].y[train_mask]: {batch['user'].y[train_mask]}")
        loss = nn.functional.cross_entropy(out, batch['user'].y[train_mask])
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=-1)
        total_correct += int((pred == batch['user'].y[train_mask]).sum())

        total_examples += train_mask.sum()
        total_loss += float(loss) * train_mask.sum()

    return (total_correct / total_examples), (total_loss / total_examples)


@torch.no_grad()
def val(val_loader):
    model.eval()

    total_examples = total_correct = total_loss = 0
    for batch in tqdm(val_loader):
        batch = batch.to(device)
        # batch_size = batch['user'].batch_size
        val_mask = batch['user'].val_mask
        out = model(batch.x_dict, batch.edge_index_dict)[val_mask]
        loss = nn.functional.cross_entropy(out, batch['user'].y[val_mask])
        # print(f"batch_size: {batch_size}")
        # print(f"val_mask: {val_mask}")
        # print(f"pred: {pred}")
        # print(f"pred[val_mask]: {pred[val_mask]}")
        # print(f"batch['user'].y: {batch['user'].y}")
        # print(f"batch['user'].y[val_mask]: {batch['user'].y[val_mask]}")
        total_examples += val_mask.sum()
        pred = out.argmax(dim=-1)
        total_correct += int((pred == batch['user'].y[val_mask]).sum())
        total_loss += loss

    return (total_correct / total_examples), (total_loss / total_examples)


@torch.no_grad()
def test(test_loader):
    model.eval()

    label = []
    out = []
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        # batch_size = batch['user'].batch_size
        test_mask = batch['user'].test_mask
        pred = model(batch.x_dict, batch.edge_index_dict)[test_mask]
        pred = pred.argmax(dim=-1)
        out.append(int(pred[0]))
        label.append(int(batch['user'].y[test_mask][0]))

    accuracy = accuracy_score(label, out)
    f1 = f1_score(label, out)
    precision = precision_score(label, out)
    recall = recall_score(label, out)

    print(f"Test: accuracy: {accuracy:.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")
    torch.save(model, rf'./saved_models/acc{accuracy:.4f}.pickle')


init_params()
best_val_acc = 0.0
best_epoch = 0
best_model = ''
lr = 0.001
for epoch in range(1, 21):
    if epoch >= 50 and epoch % 50 == 0:
        lr = 0.1 * lr
        print(f"{datetime.now()}----Current lr: {lr}.")
    train_acc, loss = train(lr)
    val_acc, val_loss = val(val_loader)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        best_model = copy.deepcopy(model.state_dict())
    print(f'Epoch: {epoch:03d}, Train_Acc: {train_acc:.4f}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Val_loss: {val_loss:.4f}')
print(f'Best val acc is: {best_val_acc:.4f}, in epoch: {best_epoch:03d}.')
model.load_state_dict(best_model)
test(test_loader)

