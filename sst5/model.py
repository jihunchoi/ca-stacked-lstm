import torch
from torch import nn

from models import caslstm
from models import utils
from models.mlp import MLP


# TODO
# - Encoder dropout (maybe weight drop or dropout w/ shared mask


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


class SSTModel(nn.Module):

    def __init__(self, num_words, num_classes,
                 word_dim, hidden_dim, enc_lstm_type, enc_bidir, enc_bidir_init,
                 enc_num_layers, enc_pool_type,
                 mlp_hidden_dim, mlp_num_layers, mlp_use_bn,
                 emb_dropout_prob, enc_dropout_prob, clf_dropout_prob):
        super().__init__()
        self.num_words = num_words
        self.num_classes = num_classes
        self.word_dim = word_dim
        self.enc_bidir = enc_bidir
        self.enc_bidir_init = enc_bidir_init
        self.enc_lstm_type = enc_lstm_type
        self.enc_pool_type = enc_pool_type
        self.hidden_dim = hidden_dim
        self.mlp_use_bn = mlp_use_bn
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
        if enc_bidir:
            self.encoder_bw = caslstm.StackedLSTM(
                lstm_type=enc_lstm_type, input_size=word_dim, hidden_size=hidden_dim,
                num_layers=enc_num_layers)
        if enc_pool_type == 'last':
            self.enc_pool = LastPool()
        elif enc_pool_type == 'max':
            self.enc_pool = MaxPool()
        else:
            raise ValueError('Unknown pool type')

        vec_dim = hidden_dim
        if enc_bidir:
            vec_dim = hidden_dim * 2

        self.mlp = MLP(input_dim=vec_dim,
                       hidden_dim=mlp_hidden_dim,
                       num_layers=mlp_num_layers,
                       use_batch_norm=mlp_use_bn,
                       dropout_prob=clf_dropout_prob)
        self.output_linear = nn.Linear(in_features=mlp_hidden_dim,
                                       out_features=num_classes)
        if mlp_use_bn:
            self.bn_mlp_input = nn.BatchNorm1d(vec_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.word_embedding.weight, mean=0, std=0.5)
        self.encoder.reset_parameters()
        if self.enc_bidir:
            self.encoder_bw.reset_parameters()
        self.mlp.reset_parameters()
        nn.init.uniform_(self.output_linear.weight.data, -0.005, 0.005)
        nn.init.constant_(self.output_linear.bias.data, val=0)
        if self.mlp_use_bn:
            self.bn_mlp_input.reset_parameters()

    def forward(self, inputs, length):
        inputs_emb = self.emb_dropout(self.word_embedding(inputs))
        (hs, cs), hx = self.encoder(inputs=inputs_emb)
        if self.enc_bidir:
            inputs_bw_emb = utils.reverse_padded_sequence(
                inputs=inputs_emb, length=length)
            if self.enc_bidir_init:
                hx_bw = hx
            (hs_bw, cs_bw), hx_bw = self.encoder_bw(inputs=inputs_bw_emb, hx=hx_bw)
            # Note that backward states are 'flipped',
            # but we will anyway apply some pooling, so leave them for now...
            hs = torch.cat([hs, hs_bw], dim=2)
        vector = self.enc_pool(inputs=hs, length=length)

        mlp_input = vector
        if self.mlp_use_bn:
            mlp_input = self.bn_mlp_input(mlp_input)
        mlp_input = self.clf_dropout(mlp_input)
        mlp_output = self.mlp(mlp_input)
        logit = self.output_linear(mlp_output)
        return logit
