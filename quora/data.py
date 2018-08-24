import logging
import os

import torch
from nltk.tokenize import word_tokenize
from torchtext import data, datasets


class TextField(data.Field):

    def __init__(self):
        super().__init__(tokenize=lambda s: s.split(),
                         lower=True, include_lengths=True, batch_first=True)


class LabelField(data.Field):

    def __init__(self):
        super().__init__(sequential=False, pad_token=None, unk_token=None)


def create_examples(path, text_field, label_field):
    examples = []
    with open(path, 'r') as f:
        for line in f:
            label, text1, text2 = line.split('\t')[:3]
            examples.append(data.Example.fromlist(
                data=(text1, text2, label),
                fields=[('text1', text_field), ('text2', text_field),
                        ('label', label_field)]))
    return examples


def load_data(root, text_field, label_field):
    quora_dir = os.path.join(root, 'quora')
    datasets_dir = os.path.join(quora_dir, 'datasets')
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
            fields=[('text1', text_field), ('text2', text_field),
                    ('label', label_field)])
        train_dataset.sort_key = lambda ex: len(ex.text1)
        valid_dataset = data.Dataset(
            examples=valid_examples,
            fields=[('text1', text_field), ('text2', text_field),
                    ('label', label_field)])
        valid_dataset.sort_key = lambda ex: len(ex.text1)
        test_dataset = data.Dataset(
            examples=test_examples,
            fields=[('text1', text_field), ('text2', text_field),
                    ('label', label_field)])
        test_dataset.sort_key = lambda ex: len(ex.text1)
    else:
        train_examples = create_examples(
            path=os.path.join(quora_dir, 'train.tsv'),
            text_field=text_field, label_field=label_field)
        valid_examples = create_examples(
            path=os.path.join(quora_dir, 'dev.tsv'),
            text_field=text_field, label_field=label_field)
        test_examples = create_examples(
            path=os.path.join(quora_dir, 'test.tsv'),
            text_field=text_field, label_field=label_field)
        train_dataset = data.Dataset(
            examples=train_examples,
            fields=[('text1', text_field), ('text2', text_field),
                    ('label', label_field)])
        train_dataset.sort_key = lambda ex: len(ex.text1)
        valid_dataset = data.Dataset(
            examples=valid_examples,
            fields=[('text1', text_field), ('text2', text_field),
                    ('label', label_field)])
        valid_dataset.sort_key = lambda ex: len(ex.text1)
        test_dataset = data.Dataset(
            examples=test_examples,
            fields=[('text1', text_field), ('text2', text_field),
                    ('label', label_field)])
        test_dataset.sort_key = lambda ex: len(ex.text1)
        os.makedirs(datasets_dir)
        torch.save(train_dataset.examples, train_examples_path)
        torch.save(valid_dataset.examples, valid_examples_path)
        torch.save(test_dataset.examples, test_examples_path)
    return train_dataset, valid_dataset, test_dataset


def trim_dataset(dataset, max_length):
    for ex in dataset.examples:
        ex.text1 = ex.text1[:max_length]
        ex.text2 = ex.text2[:max_length]
