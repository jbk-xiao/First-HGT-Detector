import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from tqdm.notebook import tqdm

from model import HGTDetector
from build_hetero_data import build_hetero_data

device = "cuda:0"

model = HGTDetector(n_cat_prop=4, n_num_prop=5, des_size=768, tweet_size=768, embedding_dimension=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
        batch_size = batch['user'].batch_size
        mask = batch['user'].train_mask
        out = model(batch.x_dict, batch.edge_index_dict)[mask]
        print(f"out[mask]: {out}")
        print(f"out[mask].argmax(-1): {out.argmax(dim=-1)}")
        print(f"batch['user'].y[mask]: {batch['user'].y[mask]}")
        loss = nn.functional.cross_entropy(out, batch['user'].y[mask].long())
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(loader):
        batch = batch.to(device, 'edge_index')
        batch_size = batch['user'].batch_size
        mask = batch['user'].val_mask
        out = model(batch.x_dict, batch.edge_index_dict)
        pred = out.argmax(dim=-1)
        print(f"batch_size: {batch_size}")
        print(f"mask: {mask}")
        print(f"pred: {pred}")
        print(f"pred[mask]: {pred[mask]}")
        print(f"batch['user'].y: {batch['user'].y}")
        print(f"batch['user'].y[mask]: {batch['user'].y[mask]}")
        total_examples += len(mask)
        total_correct += int((pred[mask] == batch['user'].y[mask]).sum())

    return total_correct / total_examples


init_params()
for epoch in range(1, 2):
    loss = train()
    val_acc = test(val_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')

