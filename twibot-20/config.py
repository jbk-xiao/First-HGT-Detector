class ModelConfig:

    def __init__(self):
        self.vocab_size = 3366013 + 1
        self.content_bow_dim = 3366013 + 1
        self.max_seq_len = 754
        self.hgt_embedding_dim = 128
        self.style_hidden_dim = 16
        self.content_hidden_dim = 112
        self.generative_emb_dim = 128
        self.num_style = 3
        self.gru_embedding_dim = 128
        self.gru_hidden_dim = 128
        self.dropout = 0.3
        # loss weights
        self.style_multitask_loss_weight = 10
        self.content_multitask_loss_weight = 3
        self.style_adversary_loss_weight = 1
        self.content_adversary_loss_weight = 0.03
        self.style_kl_lambda = 0.03
        self.content_kl_lambda = 0.03
        # kl annealing max iterations
        self.kl_anneal_iterations = 20000
        self.epsilon = 1e-8
        self.label_smoothing = 0.1

