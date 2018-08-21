def reverse_padded_sequence(inputs, length):
    inputs = inputs.clone()
    for i in range(length.shape[0]):
        inputs[i, :length[i]] = reversed(inputs[i, :length[i]])
    return inputs
