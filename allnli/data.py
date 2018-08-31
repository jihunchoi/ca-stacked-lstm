import json
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


def create_examples(path, text_field, label_field):
    examples = []
    with open(path, 'r') as f:
        for line in f:
            d = json.loads(line)
            label, premise, hypothesis = d['gold_label'], d['sentence1'], d['sentence2']
            if label == '-':
                continue
            examples.append(data.Example.fromlist(
                data=(premise, hypothesis, label),
                fields=[('premise', text_field), ('hypothesis', text_field),
                        ('label', label_field)]))
    return examples


def load_data(root, text_field, label_field):
    mnli_dir = os.path.join(root, 'mnli')
    allnli_dir = os.path.join(root, 'allnli')
    datasets_dir = os.path.join(allnli_dir, 'datasets')

    train_examples_path = os.path.join(datasets_dir, 'train_examples.pt')
    valid_examples_path = os.path.join(datasets_dir, 'valid_examples.pt')
    if os.path.exists(datasets_dir):
        logging.info(f'Loading saved dataset files from {datasets_dir}')
        train_examples = torch.load(train_examples_path)
        valid_examples = torch.load(valid_examples_path)
        train_dataset = data.Dataset(
            examples=train_examples,
            fields=[('premise', text_field), ('hypothesis', text_field),
                    ('label', label_field)])
        train_dataset.sort_key = datasets.SNLI.sort_key
        valid_dataset = data.Dataset(
            examples=valid_examples,
            fields=[('premise', text_field), ('hypothesis', text_field),
                    ('label', label_field)])
        valid_dataset.sort_key = datasets.SNLI.sort_key
    else:
        mnli_train_examples = create_examples(
            path=os.path.join(mnli_dir, 'multinli_0.9_train.jsonl'),
            text_field=text_field, label_field=label_field)
        train_dataset, valid_dataset, test_dataset = datasets.SNLI.splits(
            text_field=text_field, label_field=label_field, root=root)
        train_dataset.examples = train_dataset.examples + mnli_train_examples
        os.makedirs(datasets_dir)
        torch.save(train_dataset.examples, train_examples_path)
        torch.save(valid_dataset.examples, valid_examples_path)
    return train_dataset, valid_dataset


def trim_dataset(dataset, max_length):
    for ex in dataset.examples:
        ex.premise = ex.premise[:max_length]
        ex.hypothesis = ex.hypothesis[:max_length]
