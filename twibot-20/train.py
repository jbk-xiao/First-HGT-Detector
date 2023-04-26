import copy
import random

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import NeighborLoader, HGTLoader
from torch_geometric.data import HeteroData
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model import HGTDetector
from build_hetero_data_with_sequence import build_hetero_data

device = "cuda:0"
is_hgt_loader = False
fixed_size = 4
use_random_mask = False
remove_profiles = True


print(f"{datetime.now()}----Loading data...")
data, tweet_sequences, max_len, word_vec, _ = build_hetero_data(remove_profiles=remove_profiles, fixed_size=fixed_size)
words_size = len(word_vec)

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
        num_neighbors={key: [10] for key in data.edge_types},
        shuffle=True,
        input_nodes=('user', data['user'].train_mask),
        batch_size=256,
        num_workers=0)
    val_loader = NeighborLoader(
        data,
        num_neighbors={key: [10] for key in data.edge_types},
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

model = HGTDetector(n_cat_prop=4, n_num_prop=5, des_size=768, tweet_size=768,
                    embedding_dimension=128, word_vec=word_vec, dropout=0.3).to(device)
content_disc_params, style_disc_params, vae_and_classifiers_params, hgt_and_classification_params = model.get_params()
# ============== Define optimizers ================#
# content discriminator/adversary optimizer
content_disc_opt = torch.optim.RMSprop(
    content_disc_params, lr=1e-3)
# style discriminaot/adversary optimizer
style_disc_opt = torch.optim.RMSprop(
    style_disc_params, lr=1e-3)
# autoencoder and classifiers optimizer
vae_and_cls_opt = torch.optim.Adam(
    vae_and_classifiers_params, lr=1e-3)


@torch.no_grad()
def init_params():
    batch = next(iter(train_loader))
    batch = pad_one_batch(batch)
    batch = batch.to(device)
    model(batch, 0)


def pad_one_batch(batch):
    tweet_index = batch['tweet']['tweet_index']
    tweets_size = len(tweet_index)
    # print(f"{tweets_size} tweets a batch")
    sub_tweet_sequences = tweet_sequences[tweet_index]
    seq_lengths = batch['tweet']['seq_length']
    pad_tweet_sequences = np.ones((tweets_size, max_len)) * (words_size - 1)
    content_bow = torch.zeros(tweets_size, 500)
    for i in range(tweets_size):
        for word in sub_tweet_sequences[i]:
            if word < 500:
                content_bow[i][word] += 1
        pad_tweet_sequences[i][0:seq_lengths[i]] = sub_tweet_sequences[i]
    sub_tweet_sequences = torch.tensor(pad_tweet_sequences).int()
    style_labels = batch['tweet']['style_label']

    batch_data = HeteroData(
        {
            'user': {
                'x': batch['user']['x'],
                'y': batch['user']['y'],
                'train_mask': batch['user']['train_mask'],
                'val_mask': batch['user']['val_mask'],
                'test_mask': batch['user']['test_mask']
            },
            'tweet': {
                'x': batch['tweet']['x'],
                'sequence': sub_tweet_sequences,
                'seq_length': seq_lengths,
                'style_label': style_labels,
                'content_bow': content_bow
            }
        }
    )
    batch_data.edge_index_dict = batch.edge_index_dict
    return batch_data


def train(lr):
    model.train()
    hgt_and_classification_optimizer = torch.optim.AdamW(hgt_and_classification_params, lr=lr, weight_decay=1e-5)
    total_examples = total_correct = total_loss = 0
    total_content_disc_loss = total_style_disc_loss = total_vae_and_classifier_loss = 0
    for iteration, batch in enumerate(tqdm(train_loader)):
        hgt_and_classification_optimizer.zero_grad()
        if use_random_mask:
            random_mask = random.randrange(0, 9)
            batch['user'].x[:, random_mask] = 0
            random_mask = random.choices(range(768), k=20)
            batch['user'].x[:, [x + 9 for x in random_mask]] = 0
            # random_mask = random.choices(range(768), k=20)
            # batch['tweet'].x[:, random_mask] = 0

        batch = pad_one_batch(batch)
        batch = batch.to(device)

        train_mask = batch['user'].train_mask
        content_disc_loss, style_disc_loss, vae_and_classifier_loss, out = model(batch, iteration)
        out = out[train_mask]
        # out = model(batch.x_dict, batch.edge_index_dict)[train_mask]
        # ============== Update Adversary/Discriminator parameters ===========#
        # update content discriminator parametes
        # we need to retain the computation graph so that discriminator predictions are
        # not freed as we need them to calculate entropy.
        # Note that even detaching the discriminator branch won't help us since it will
        # be freed and delete all the intermediary values(predictions, in our case).
        # Hence, with no access to this branch we can't backprop the entropy loss
        content_disc_loss.backward(retain_graph=True)
        content_disc_opt.step()
        content_disc_opt.zero_grad()
        total_content_disc_loss += float(content_disc_loss)

        # update style discriminator parameters
        style_disc_loss.backward(retain_graph=True)
        style_disc_opt.step()
        style_disc_opt.zero_grad()
        total_style_disc_loss += float(style_disc_loss)

        # =============== Update VAE and classifier parameters ===============#
        vae_and_classifier_loss.backward()
        vae_and_cls_opt.step()
        vae_and_cls_opt.zero_grad()
        total_vae_and_classifier_loss += float(vae_and_classifier_loss)

        # loss for classification
        loss = nn.functional.cross_entropy(out, batch['user'].y[train_mask])
        loss.backward()
        hgt_and_classification_optimizer.step()
        pred = out.argmax(dim=-1)
        total_correct += int((pred == batch['user'].y[train_mask]).sum())

        total_examples += train_mask.sum()
        total_loss += float(loss) * train_mask.sum()
    print(f"DRL loss: total_content_disc_loss: {total_content_disc_loss}, total_style_disc_loss: {total_style_disc_loss}"
          f", total_vae_and_classifier_loss: {total_vae_and_classifier_loss}.")

    return (total_correct / total_examples), (total_loss / total_examples)


@torch.no_grad()
def val(val_loader):
    model.eval()

    total_examples = total_correct = total_loss = 0
    for iteration, batch in enumerate(tqdm(val_loader)):
        batch = pad_one_batch(batch)
        batch = batch.to(device)
        # batch_size = batch['user'].batch_size
        val_mask = batch['user'].val_mask
        _, _, _, out = model(batch, iteration)
        out = out[val_mask]
        # out = model(batch.x_dict, batch.edge_index_dict)[val_mask]
        loss = nn.functional.cross_entropy(out, batch['user'].y[val_mask])
        total_examples += val_mask.sum()
        pred = out.argmax(dim=-1)
        total_correct += int((pred == batch['user'].y[val_mask]).sum())
        total_loss += float(loss) * val_mask.sum()

    return (total_correct / total_examples), (total_loss / total_examples)


@torch.no_grad()
def test(test_loader):
    model.eval()

    label = []
    out = []
    for iteration, batch in enumerate(tqdm(test_loader)):
        batch = pad_one_batch(batch)
        batch = batch.to(device)
        test_mask = batch['user'].test_mask
        _, _, _, pred = model(batch, iteration)
        pred = pred[test_mask]
        # pred = model(batch.x_dict, batch.edge_index_dict)[test_mask]
        pred = pred.argmax(dim=-1)
        out.append(int(pred[0]))
        label.append(int(batch['user'].y[test_mask][0]))

    accuracy = accuracy_score(label, out)
    f1 = f1_score(label, out)
    precision = precision_score(label, out)
    recall = recall_score(label, out)

    print(f"Test: accuracy: {accuracy:.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")
    torch.save(model, rf'./saved_models/drl-acc{accuracy:.4f}.pickle')


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
