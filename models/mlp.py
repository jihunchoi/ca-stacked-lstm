from collections import OrderedDict

from torch import nn
from torch.nn import init


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers,
                 use_batch_norm, dropout_prob):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.dropout_prob = dropout_prob

        sequential_layers = OrderedDict()
        for i in range(num_layers):
            if i == 0:
                in_features = input_dim
            else:
                in_features = hidden_dim
            sequential_layers[f'linear_{i}'] = nn.Linear(
                in_features=in_features, out_features=hidden_dim)
            sequential_layers[f'relu_{i}'] = nn.ReLU()
            if self.use_batch_norm:
                sequential_layers[f'bn_{i}'] = nn.BatchNorm1d(hidden_dim)
            sequential_layers[f'dropout_{i}'] = nn.Dropout(dropout_prob)
        self.net = nn.Sequential(sequential_layers)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_layers):
            linear_layer = getattr(self.net, f'linear_{i}')
            init.kaiming_normal_(linear_layer.weight.data)
            init.constant_(linear_layer.bias.data, val=0)
            if self.use_batch_norm:
                bn_layer = getattr(self.net, f'bn_{i}')
                bn_layer.reset_parameters()

    def forward(self, input):
        return self.net(input)
