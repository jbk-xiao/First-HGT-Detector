import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm, trange

from text_dataset import TextDataset
from textual_drl_model import AdversarialVAE
from config import ModelConfig

model_config = ModelConfig()
content_bow_dim = model_config.content_bow_dim

device = "cuda:0"

max_epochs = 20

tmp_files_root = r"./preprocess/tmp-files"

word_vec = np.load(rf"{tmp_files_root}/less_vec.npy")
word_vec = torch.tensor(word_vec)[0:content_bow_dim - 1]
blank_vec = torch.zeros((1, word_vec.shape[-1]))
word_vec = torch.cat((word_vec, blank_vec), dim=0)

vae_model = AdversarialVAE(word_vec).to(device)

train_set = TextDataset('train')
train_loader = DataLoader(train_set, batch_size=256)
content_discriminator_params, style_discriminator_params, vae_and_classifier_params = vae_model.get_params()
# ============== Define optimizers ================#
# content discriminator/adversary optimizer
content_disc_opt = torch.optim.RMSprop(content_discriminator_params, lr=1e-3)
# style discriminator/adversary optimizer
style_disc_opt = torch.optim.RMSprop(style_discriminator_params, lr=1e-3)
# autoencoder and classifiers optimizer
vae_and_cls_opt = torch.optim.Adam(vae_and_classifier_params, lr=1e-3)
print("Training started!")
total_loss = 0
for epoch in trange(max_epochs, desc="Epoch"):
    total_content_disc_loss = total_style_disc_loss = total_vae_and_classifier_loss = 0
    for iteration, batch in enumerate(tqdm(train_loader)):
        sequences, seq_lengths, style_labels, content_bow = batch
        content_disc_loss, style_disc_loss, vae_and_classifier_loss, _ = vae_model(
            sequences.to(device), seq_lengths.to(device), style_labels.to(device), content_bow.to(device), iteration + 1
        )
        # ============== Update Adversary/Discriminator parameters ===========#
        # update content discriminator parameters
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
    print(f"total_content_disc_loss: {total_content_disc_loss}, total_style_disc_loss: {total_style_disc_loss}"
          f", total_vae_and_classifier_loss: {total_vae_and_classifier_loss}.")
    total_loss = total_vae_and_classifier_loss

torch.save(vae_model, rf"saved_models/vae-{max_epochs}-{total_loss:.6f}.pickle")
torch.save(
    {
        'content_disc': content_disc_opt.state_dict(),
        'style_disc': style_disc_opt.state_dict(),
        'vae_and_cls': vae_and_cls_opt.state_dict()
    },
    rf"saved_models/vae_opt-{max_epochs}-{total_loss:.6f}.pickle"
)
