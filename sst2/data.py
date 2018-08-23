import logging
import os

import torch
from nltk.tokenize import word_tokenize
from torchtext import data, datasets


class TextField(data.Field):

    def __init__(self):
        super().__init__(tokenize=word_tokenize,
                         lower=True, include_lengths=True, batch_first=True)


class LabelField(data.Field):

    def __init__(self):
        super().__init__(sequential=False, pad_token=None, unk_token=None)


def load_data(root, text_field, label_field):
    sst2_dir = os.path.join(root, 'sst', 'sst2')
    datasets_dir = os.path.join(sst2_dir, 'datasets')
    train_examples_path = os.path.join(datasets_dir, 'train_examples.pt')
    valid_examples_path = os.path.join(datasets_dir, 'valid_examples.pt')
    test_examples_path = os.path.join(datasets_dir, 'test_examples.pt')
    if os.path.exists(datasets_dir):
        logging.info(f'Loading saved dataset files from {datasets_dir}')
        train_examples = torch.load(train_examples_path)
        valid_examples = torch.load(valid_examples_path)
        test_examples = torch.load(test_examples_path)
        train_dataset = data.Dataset(
            examples=train_examples,
            fields=[('text', text_field), ('label', label_field)])
        train_dataset.sort_key = datasets.SST.sort_key
        valid_dataset = data.Dataset(
            examples=valid_examples,
            fields=[('text', text_field), ('label', label_field)])
        valid_dataset.sort_key = datasets.SST.sort_key
        test_dataset = data.Dataset(
            examples=test_examples,
            fields=[('text', text_field), ('label', label_field)])
        test_dataset.sort_key = datasets.SST.sort_key
    else:
        train_dataset, valid_dataset, test_dataset = datasets.SST.splits(
            text_field=text_field, label_field=label_field, root=root,
            train_subtrees=True, fine_grained=False,
            filter_pred=lambda ex: ex.label != 'neutral')
        os.makedirs(datasets_dir)
        torch.save(train_dataset.examples, train_examples_path)
        torch.save(valid_dataset.examples, valid_examples_path)
        torch.save(test_dataset.examples, test_examples_path)
    return train_dataset, valid_dataset, test_dataset
