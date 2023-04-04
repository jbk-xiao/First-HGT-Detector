import copy

import torch
from torch import nn
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model import HGTDetector
from build_hetero_data import build_hetero_data

device = "cuda:0"

model = HGTDetector(n_cat_prop=4, n_num_prop=5, des_size=768, tweet_size=768, embedding_dimension=128, dropout=0.3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"{datetime.now()}----Loading data...")
data = build_hetero_data()

train_loader = NeighborLoader(
    data,
    num_neighbors={key: [-1] for key in data.edge_types},
    shuffle=True,
    input_nodes=('user', data['user'].train_mask),
    batch_size=128,
    num_workers=0)
val_loader = NeighborLoader(
    data,
    num_neighbors={key: [-1] for key in data.edge_types},
    shuffle=True,
    input_nodes=('user', data['user'].val_mask),
    batch_size=128,
    num_workers=0)
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


print(f"{datetime.now()}----Data loaded.")


@torch.no_grad()
def init_params():
    batch = next(iter(train_loader))
    batch = batch.to(device)
    model(batch.x_dict, batch.edge_index_dict)


def train():
    model.train()

    total_examples = total_correct = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        # batch_size = batch['user'].batch_size
        mask = batch['user'].train_mask
        out = model(batch.x_dict, batch.edge_index_dict)[mask]
        pred = out.argmax(dim=-1)
        total_correct += int((pred == batch['user'].y[mask]).sum())
        # print(f"out[mask]: {out}")
        # print(f"out[mask].argmax(-1): {out.argmax(dim=-1)}")
        # print(f"batch['user'].y[mask]: {batch['user'].y[mask]}")
        loss = nn.functional.cross_entropy(out, batch['user'].y[mask])
        loss.backward()
        optimizer.step()

        total_examples += mask.sum()
        total_loss += float(loss) * mask.sum()

    return (total_correct / total_examples), (total_loss / total_examples)


@torch.no_grad()
def val(loader):
    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(loader):
        batch = batch.to(device)
        # batch_size = batch['user'].batch_size
        mask = batch['user'].val_mask
        out = model(batch.x_dict, batch.edge_index_dict)
        pred = out.argmax(dim=-1)[mask]
        # print(f"batch_size: {batch_size}")
        # print(f"mask: {mask}")
        # print(f"pred: {pred}")
        # print(f"pred[mask]: {pred[mask]}")
        # print(f"batch['user'].y: {batch['user'].y}")
        # print(f"batch['user'].y[mask]: {batch['user'].y[mask]}")
        total_examples += mask.sum()
        total_correct += int((pred == batch['user'].y[mask]).sum())

    return total_correct / total_examples


@torch.no_grad()
def test(test_data):
    model.eval()
    test_data.to(device)
    out = model(test_data.x_dict, test_data.edge_index_dict).argmax(dim=-1)
    out = out.cpu()
    label = test_data['user'].y
    label = label.cpu()
    accuracy = accuracy_score(label, out)
    f1 = f1_score(label, out)
    precision = precision_score(label, out)
    recall = recall_score(label, out)

    print(f"Test, accuracy: {accuracy:.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")
    torch.save(model, rf'./saved_models/acc{accuracy:.4f}.pickle')


init_params()
best_val_acc = 0.0
best_epoch = 0
best_model = ''
for epoch in range(1, 21):
    train_acc, loss = train()
    val_acc = val(val_loader)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        best_model = copy.deepcopy(model.state_dict())
    print(f'Epoch: {epoch:03d}, Train_Acc: {train_acc:.4f}, Loss: {loss:.4f}, Val: {val_acc:.4f}')
print(f'Best val acc is: {best_val_acc:.4f}, in epoch: {best_epoch:03d}.')
model.load_state_dict(best_model)
test(test_data)

