import torch
from torch import nn
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model import HGTDetector
from build_hetero_data import build_hetero_data

device = "cuda:0"

model = HGTDetector(n_cat_prop=4, n_num_prop=5, des_size=768, tweet_size=768, embedding_dimension=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"{datetime.now()}----Loading data...")
data = build_hetero_data().to(device, 'x', 'y')

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
).to("cuda:0")

print(f"{datetime.now()}----Data loaded.")


@torch.no_grad()
def init_params():
    batch = next(iter(train_loader))
    batch = batch.to(device, "edge_index")
    model(batch.x_dict, batch.edge_index_dict)


def train():
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device, 'edge_index')
        # batch_size = batch['user'].batch_size
        # mask = batch['user'].train_mask
        out = model(batch.x_dict, batch.edge_index_dict)
        # print(f"out[mask]: {out}")
        # print(f"out[mask].argmax(-1): {out.argmax(dim=-1)}")
        # print(f"batch['user'].y[mask]: {batch['user'].y[mask]}")
        loss = nn.functional.cross_entropy(out, batch['user'].y.long())
        loss.backward()
        optimizer.step()

        total_examples += len(out)
        total_loss += float(loss) * len(out)

    return total_loss / total_examples


@torch.no_grad()
def val(loader):
    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(loader):
        batch = batch.to(device, 'edge_index')
        # batch_size = batch['user'].batch_size
        # mask = batch['user'].val_mask
        out = model(batch.x_dict, batch.edge_index_dict)
        pred = out.argmax(dim=-1)
        # print(f"batch_size: {batch_size}")
        # print(f"mask: {mask}")
        # print(f"pred: {pred}")
        # print(f"pred[mask]: {pred[mask]}")
        # print(f"batch['user'].y: {batch['user'].y}")
        # print(f"batch['user'].y[mask]: {batch['user'].y[mask]}")
        total_examples += len(out)
        total_correct += int((pred == batch['user'].y).sum())

    return total_correct / total_examples


@torch.no_grad()
def test(test_data):
    model.to("cuda:0")
    model.eval()

    out = model(test_data.x_dict, test_data.edge_index_dict).argmax(dim=-1)
    label = test_data['user'].y
    accuracy = accuracy_score(label, out)
    f1 = f1_score(label, out)
    precision = precision_score(label, out)
    recall = recall_score(label, out)

    print(f"Test, accuracy: {accuracy:.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")


init_params()
for epoch in range(1, 2):
    loss = train()
    val_acc = val(val_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')
test(test_data)

