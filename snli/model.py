import torch
from torch import nn

from models import caslstm
from models import matching
from models.mlp import MLP


class LastPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs, length):
        """
        input: (B, T, H)
        length: (B,)
        """

        batch_size = length.shape[0]
        last_vector = inputs[range(batch_size), length - 1]
        return last_vector


class MaxPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs, length):
        batch_size, max_length = inputs.shape[:2]
        range_ = torch.arange(max_length).unsqueeze(0).to(length)  # (1, T)
        length = length.unsqueeze(1)  # (B, 1)
        mask = length.gt(range_).unsqueeze(2)  # (B, T, 1)
        inputs = inputs.masked_fill(mask=~mask, value=float('-inf'))
        return inputs.max(1)[0]


class SNLIModel(nn.Module):

    def __init__(self, num_words, num_classes,
                 word_dim, hidden_dim, enc_lstm_type, enc_num_layers, enc_pool_type,
                 mlp_hidden_dim, mlp_num_layers, mlp_use_bn, matching_type,
                 emb_dropout_prob, enc_dropout_prob, clf_dropout_prob):
        super().__init__()
        self.num_words = num_words
        self.num_classes = num_classes
        self.word_dim = word_dim
        self.enc_lstm_type = enc_lstm_type
        self.enc_pool_type = enc_pool_type
        self.hidden_dim = hidden_dim
        self.mlp_use_bn = mlp_use_bn
        self.matching_type = matching_type
        self.emb_dropout_prob = emb_dropout_prob
        self.enc_dropout_prob = enc_dropout_prob
        self.mlp_dropout_prob = clf_dropout_prob

        self.emb_dropout = nn.Dropout(emb_dropout_prob)
        self.clf_dropout = nn.Dropout(clf_dropout_prob)
        self.word_embedding = nn.Embedding(
            num_embeddings=num_words, embedding_dim=word_dim)
        self.encoder = caslstm.StackedLSTM(
            lstm_type=enc_lstm_type, input_size=word_dim, hidden_size=hidden_dim,
            num_layers=enc_num_layers)
        if enc_pool_type == 'last':
            self.enc_pool = LastPool()
        elif enc_pool_type == 'max':
            self.enc_pool = MaxPool()
        else:
            raise ValueError('Unknown pool type')

        if matching_type == 'heuristic':
            self.matching = matching.HeuristicMatching(hidden_dim)
        elif matching_type == 'heuristic2':
            self.matching = matching.HeuristicMatching2(hidden_dim)
        else:
            raise ValueError('Unknown matching type')

        mlp_input_dim = self.matching.output_dim

        self.mlp = MLP(input_dim=mlp_input_dim,
                       hidden_dim=mlp_hidden_dim,
                       num_layers=mlp_num_layers,
                       use_batch_norm=mlp_use_bn,
                       dropout_prob=clf_dropout_prob)
        self.output_linear = nn.Linear(in_features=mlp_hidden_dim,
                                       out_features=num_classes)
        if mlp_use_bn:
            self.bn_mlp_input = nn.BatchNorm1d(mlp_input_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.word_embedding.weight, mean=0, std=0.01)
        self.encoder.reset_parameters()
        self.mlp.reset_parameters()
        nn.init.uniform_(self.output_linear.weight.data, -0.005, 0.005)
        nn.init.constant_(self.output_linear.bias.data, val=0)
        self.matching.reset_parameters()
        if self.mlp_use_bn:
            self.bn_mlp_input.reset_parameters()

    def forward(self, pre_inputs, pre_length, hyp_inputs, hyp_length):
        pre_inputs_emb = self.emb_dropout(self.word_embedding(pre_inputs))
        hyp_inputs_emb = self.emb_dropout(self.word_embedding(hyp_inputs))
        (pre_hs, pre_cs), pre_hx = self.encoder(inputs=pre_inputs_emb)
        (hyp_hs, hyp_cs), hyp_hx = self.encoder(inputs=hyp_inputs_emb)
        pre_vector = self.enc_pool(inputs=pre_hs, length=pre_length)
        hyp_vector = self.enc_pool(inputs=hyp_hs, length=hyp_length)

        mlp_input = self.matching(s1=pre_vector, s2=hyp_vector)
        if self.mlp_use_bn:
            mlp_input = self.bn_mlp_input(mlp_input)
        mlp_input = self.clf_dropout(mlp_input)
        mlp_output = self.mlp(mlp_input)
        logit = self.output_linear(mlp_output)
        return logit
