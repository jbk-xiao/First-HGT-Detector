import torch

from tqdm import tqdm

from text_dataset_per_user import TextDataset

device = 'cpu'
vae_model_file = "vae-2-5368.065876.pickle"
vae_model = torch.load(rf'saved_models/{vae_model_file}').to(device)

generate_dataset = TextDataset('generate')
gen_iter = generate_dataset.get_iter()

tweet_embs = []
for tweet_sequence, seq_length in tqdm(gen_iter, desc='Generating tweet emb...', ncols=generate_dataset.__len__()):
    content_emb, style_emb = vae_model.get_style_content_emb(
        tweet_sequence.unsqueeze(0).to(device), seq_length.unsqueeze(0).to(device)
    )
    tweet_emb = torch.concat((content_emb, style_emb), dim=1)
    tweet_embs.append(tweet_emb)

tweet_embs = torch.concat(tweet_embs, dim=0)
torch.save(tweet_embs, rf"./preprocess/tmp-files/{vae_model_file}-tweet_embs.pt")
