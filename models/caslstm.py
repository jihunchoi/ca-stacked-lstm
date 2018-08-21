import torch
from torch import nn


class CALSTMCell(nn.Module):

    """Cell-aware LSTM cell."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=2 * hidden_size,
                                out_features=5 * hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, val=0)

    def forward(self, input, hx=None):
        """
        input: (h, c)
        hx: (h, c)
        """

        h_lower, c_lower = input
        if hx is None:
            device = h_lower.device
            zero_state = torch.zeros(h_lower.shape[0], self.hidden_size, device=device)
            hx = (zero_state, zero_state)
        h_prev, c_prev = hx
        h_cat = torch.cat([h_lower, h_prev], dim=1)
        lstm_matrix = self.linear(h_cat)
        i, f_prev, f_lower, u, o = lstm_matrix.chunk(chunks=5, dim=1)
        c = (c_prev * (f_prev + 1).sigmoid()
             + c_lower * (f_lower + 1).sigmoid()
             + u.tanh() * i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c


class CALSTM(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.cell = CALSTMCell(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.cell.reset_parameters()

    def forward(self, inputs, length=None, hx=None):
        """
        inputs: A tuple of (h, c), each of which is in shape (B, T, Dh).
        """

        batch_size, max_length = inputs[0].shape[:2]
        hs = []
        cs = []
        for t in range(max_length):
            hx = self.cell(input=(inputs[0][:, t], inputs[1][:, t]), hx=hx)
            hs.append(hx[0])
            cs.append(hx[1])
        hs = torch.stack(hs, dim=1)
        cs = torch.stack(cs, dim=1)
        if length is not None:
            h = hs[range(batch_size), length - 1]
            c = cs[range(batch_size), length - 1]
            hx = (h, c)
        return (hs, cs), hx


class StatefulLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.cell.weight_hh)
        nn.init.kaiming_normal_(self.cell.weight_ih)
        nn.init.constant_(self.cell.bias_hh, val=0)
        nn.init.constant_(self.cell.bias_ih, val=0)
        nn.init.constant_(self.cell.bias_ih.chunk(4)[1], val=1)

    def forward(self, inputs, length=None, hx=None):
        """
        inputs: (B, T, Di).
        """

        hs = []
        cs = []
        if isinstance(inputs, tuple):
            # Only use hidden states as input
            inputs = inputs[0]
        batch_size, max_length = inputs.shape[:2]
        for t in range(max_length):
            hx = self.cell(input=inputs[:, t], hx=hx)
            hs.append(hx[0])
            cs.append(hx[1])
        hs = torch.stack(hs, dim=1)
        cs = torch.stack(cs, dim=1)
        if length is not None:
            h = hs[range(batch_size), length - 1]
            c = cs[range(batch_size), length - 1]
            hx = (h, c)
        return (hs, cs), hx


class StackedLSTM(nn.Module):

    def __init__(self, lstm_type, input_size, hidden_size, num_layers):
        super().__init__()
        assert lstm_type != 'ca' or num_layers != 1, (
            'CA Stacked LSTM is equivalent to the plain LSTM when num_layers == 1')

        self.num_layers = num_layers
        if lstm_type == 'plain':
            self.lstms = nn.ModuleList()
            for i in range(num_layers):
                layer_input_size = input_size if i == 0 else hidden_size
                layer = StatefulLSTM(input_size=layer_input_size,
                                     hidden_size=hidden_size)
                self.lstms.append(layer)
        elif lstm_type == 'ca':
            self.lstms = nn.ModuleList()
            self.lstms.append(StatefulLSTM(input_size=input_size,
                                           hidden_size=hidden_size))
            for i in range(1, num_layers):
                self.lstms.append(CALSTM(hidden_size))
        else:
            raise ValueError('Unknown LSTM type')
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.lstms:
            layer.reset_parameters()

    def forward(self, inputs, length=None, hx=None):
        layer_inputs = inputs
        state = []
        if hx is None:
            hx = [None] * self.num_layers
        for layer, prev_state in zip(self.lstms, hx):
            layer_outputs, layer_state = layer(
                inputs=layer_inputs, length=length, hx=prev_state)
            layer_inputs = layer_outputs
            state.append(layer_state)
        return layer_inputs, state
