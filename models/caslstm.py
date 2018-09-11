import math

import torch
from torch import nn


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        lstm_matrix = (torch.matmul(input, self.weight_ih.t())
                       + torch.matmul(hx[0], self.weight_hh.t()))
        if self.bias:
            lstm_matrix = lstm_matrix + self.bias_ih + self.bias_hh
        i, f, g, o = lstm_matrix.chunk(chunks=4, dim=1)
        c = i.sigmoid()*g.tanh() + f.sigmoid()*hx[1]
        h = o.sigmoid() * c.tanh()
        return (h, c), (f.sigmoid(), o.sigmoid())


class CALSTMCell(nn.Module):

    """Cell-aware LSTM cell."""

    def __init__(self, hidden_size, h_lower_proj=None, fuse_type=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.fuse_type = fuse_type
        if h_lower_proj is not None:
            self.linear_hh = nn.Linear(in_features=hidden_size,
                                       out_features=5 * hidden_size)
            self.linear_ih = nn.Linear(in_features=hidden_size,
                                       out_features=4 * hidden_size, bias=False)
        else:
            self.linear = nn.Linear(in_features=2 * hidden_size,
                                    out_features=5 * hidden_size)
        if fuse_type == 'param':
            self.forget_weight = nn.Parameter(torch.FloatTensor(hidden_size))
        self.h_lower_proj = h_lower_proj
        self.reset_parameters()

    def extra_repr(self):
        return f'fuse_type={self.fuse_type}'

    def reset_parameters(self):
        if self.h_lower_proj is not None:
            nn.init.orthogonal_(self.linear_hh.weight)
            nn.init.orthogonal_(self.linear_ih.weight)
            nn.init.constant_(self.linear_hh.bias, val=0)
        else:
            nn.init.orthogonal_(self.linear.weight)
            nn.init.constant_(self.linear.bias, val=0)
        if self.fuse_type == 'param':
            nn.init.normal_(self.forget_weight, mean=0, std=0.01)

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
        if self.h_lower_proj is not None:
            ih, fh_prev, fh_lower, uh, oh = (
                self.linear_hh(h_prev).chunk(chunks=5, dim=1))
            il, fl_prev, ul, ol = self.linear_ih(h_lower).chunk(chunks=4, dim=1)
            i = ih + il
            f_prev = fh_prev + fl_prev
            f_lower = fh_lower + self.h_lower_proj(h_lower)  # Trick.
            u = uh + ul
            o = oh + ol
        else:
            h_cat = torch.cat([h_lower, h_prev], dim=1)
            lstm_matrix = self.linear(h_cat)
            i, f_prev, f_lower, u, o = lstm_matrix.chunk(chunks=5, dim=1)
        if self.fuse_type == 'add':
            c = (c_prev * (f_prev + 1).sigmoid()
                 + c_lower * (f_lower + 1).sigmoid()
                 + u.tanh() * i.sigmoid())
        elif self.fuse_type == 'avg':
            c = (0.5 * (c_prev * (f_prev + 1).sigmoid()
                       + c_lower * (f_lower + 1).sigmoid())
                 + u.tanh() * i.sigmoid())
        elif self.fuse_type == 'param':
            weight = self.forget_weight.sigmoid()
            c = (weight * (c_prev * (f_prev + 1).sigmoid())
                 + (1 - weight) * (c_lower * (f_lower + 1).sigmoid())
                 + u.tanh() * i.sigmoid())
        else:
            lam = 1 - float(self.fuse_type)
            c = (lam * (c_prev * (f_prev + 1).sigmoid())
                 + (1 - lam) * (c_lower * (f_lower + 1).sigmoid())
                 + u.tanh() * i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return (h, c), ((f_prev + 1).sigmoid(), (f_lower + 1).sigmoid(), o.sigmoid())


class CALSTM(nn.Module):

    def __init__(self, hidden_size, h_lower_proj=None, fuse_type=False):
        super().__init__()
        self.cell = CALSTMCell(hidden_size=hidden_size, h_lower_proj=h_lower_proj,
                               fuse_type=fuse_type)
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
            hx, _ = self.cell(input=(inputs[0][:, t], inputs[1][:, t]), hx=hx)
            hs.append(hx[0])
            cs.append(hx[1])
        hs = torch.stack(hs, dim=1)
        cs = torch.stack(cs, dim=1)
        if length is not None:
            h = hs[range(batch_size), length - 1]
            c = cs[range(batch_size), length - 1]
            hx = (h, c)
        return (hs, cs), hx


class HighwayLSTMCell(nn.Module):

    """Cell-aware LSTM cell."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=2 * hidden_size,
                                out_features=4 * hidden_size)
        self.linear_xd = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.peephole_ci = nn.Parameter(torch.FloatTensor(hidden_size))
        self.peephole_cf = nn.Parameter(torch.FloatTensor(hidden_size))
        self.peephole_co = nn.Parameter(torch.FloatTensor(hidden_size))
        self.peephole_cd = nn.Parameter(torch.FloatTensor(hidden_size))
        self.peephole_ld = nn.Parameter(torch.FloatTensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, val=0)
        nn.init.kaiming_normal_(self.linear_xd.weight)
        nn.init.constant_(self.linear_xd.bias, val=0)
        nn.init.normal_(self.peephole_ci, mean=0, std=0.01)
        nn.init.normal_(self.peephole_cf, mean=0, std=0.01)
        nn.init.normal_(self.peephole_co, mean=0, std=0.01)
        nn.init.normal_(self.peephole_cd, mean=0, std=0.01)
        nn.init.normal_(self.peephole_ld, mean=0, std=0.01)

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
        i, f, o, u = lstm_matrix.chunk(chunks=4, dim=1)
        i = i + self.peephole_ci*c_prev
        f = f + self.peephole_cf*c_prev
        o = f + self.peephole_co*c_prev
        d = self.linear_xd(h_lower) + self.peephole_cd*c_prev + self.peephole_ld*c_lower
        c = d.sigmoid()*c_lower + f.sigmoid()*c_prev + i.sigmoid()*u.tanh()
        h = o.sigmoid() * c.tanh()
        return h, c


class HighwayLSTM(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.cell = HighwayLSTMCell(hidden_size=hidden_size)
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
        self.cell = LSTMCell(input_size=input_size, hidden_size=hidden_size)
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
            hx = self.cell(input=inputs[:, t], hx=hx)[0]
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

    def __init__(self, lstm_type, input_size, hidden_size, num_layers,
                 shared_h_lower_proj=False, fuse_type=False):
        super().__init__()
        assert lstm_type != 'ca' or num_layers != 1, (
            'CA Stacked LSTM is equivalent to the plain LSTM when num_layers == 1')
        assert lstm_type != 'plain' or not shared_h_lower_proj, (
            'shared_h_lower_proj is ignored when lstm_type == plain')

        self.lstm_type = lstm_type
        self.num_layers = num_layers
        self.shared_h_lower_proj = shared_h_lower_proj
        if lstm_type == 'plain':
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True)
        elif lstm_type == 'ca':
            self.lstms = nn.ModuleList()
            self.lstms.append(StatefulLSTM(input_size=input_size,
                                           hidden_size=hidden_size))
            for i in range(1, num_layers):
                h_lower_proj = None
                if shared_h_lower_proj:
                    h_lower_proj = nn.Linear(
                        in_features=hidden_size, out_features=hidden_size, bias=False)
                self.lstms.append(CALSTM(hidden_size=hidden_size,
                                         h_lower_proj=h_lower_proj, fuse_type=fuse_type))
        elif lstm_type == 'highway':
            self.lstms = nn.ModuleList()
            self.lstms.append(StatefulLSTM(input_size=input_size,
                                           hidden_size=hidden_size))
            for i in range(1, num_layers):
                self.lstms.append(HighwayLSTM(hidden_size=hidden_size))
        else:
            raise ValueError('Unknown LSTM type')
        self.reset_parameters()

    def extra_repr(self):
        return f'shared_h_lower_proj={self.shared_h_lower_proj}'

    def reset_parameters(self):
        if self.lstm_type != 'plain':
            for layer in self.lstms:
                layer.reset_parameters()
        else:
            self.lstm.reset_parameters()
        if self.shared_h_lower_proj:
            nn.init.orthogonal_(self.lstms[1].cell.h_lower_proj.weight)

    def forward(self, inputs, length=None, hx=None):
        if self.lstm_type != 'plain':
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
        else:
            outputs, state = self.lstm(input=inputs, hx=hx)
            return (outputs, None), state
