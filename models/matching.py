import torch
from torch import nn


class HeuristicMatching(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 4 * input_dim

    def reset_parameters(self):
        pass

    def forward(self, s1, s2):
        heuristic_features = [s1, s2, (s1 - s2).abs(), s1 * s2]
        return torch.cat(heuristic_features, dim=1)


class HeuristicMatching2(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 5 * input_dim

    def reset_parameters(self):
        pass

    def forward(self, s1, s2):
        heuristic_features = [s1, s2, (s1 - s2).abs(), s1 - s2, s1 * s2]
        return torch.cat(heuristic_features, dim=1)
