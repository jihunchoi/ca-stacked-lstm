import torch


def reverse_padded_sequence(inputs, length):
    batch_size, max_length = inputs.shape[:2]
    lengths = length.cpu().tolist()
    ind = [list(range(l - 1, -1, -1)) + list(range(l, max_length))
           for l in lengths]
    ind = torch.tensor(ind, device=inputs.device)
    for d in range(2, inputs.dim()):
        ind = ind.unsqueeze(d)
    ind = ind.expand_as(inputs)
    reversed_inputs = torch.gather(inputs, dim=1, index=ind)
    return reversed_inputs
