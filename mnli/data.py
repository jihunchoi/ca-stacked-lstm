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


class PairIDField(data.Field):

    def __init__(self):
        super().__init__(sequential=False, pad_token=None, unk_token=None,
                         use_vocab=False)


def create_examples(path, text_field, label_field, pair_id_field):
    examples = []
    with open(path, 'r') as f:
        for line in f:
            d = json.loads(line)
            pair_id, genre, label, premise, hypothesis = (
                d['pairID'], d['genre'], d['gold_label'], d['sentence1'], d['sentence2']
            )
            if label == '-':
                continue
            examples.append(data.Example.fromlist(
                data=(premise, hypothesis, label, pair_id, genre),
                fields=[('premise', text_field), ('hypothesis', text_field),
                        ('label', label_field), ('pair_id', pair_id_field)]))
    return examples


def load_data(root, text_field, label_field):
    mnli_dir = os.path.join(root, 'mnli')
    datasets_dir = os.path.join(mnli_dir, 'datasets')
    pair_id_field = PairIDField()

    train_examples_path = os.path.join(datasets_dir, 'train_examples.pt')
    valid_m_examples_path = os.path.join(datasets_dir, 'valid_m_examples.pt')
    valid_mm_examples_path = os.path.join(datasets_dir, 'valid_mm_examples.pt')
    test_m_examples_path = os.path.join(datasets_dir, 'test_m_examples.pt')
    test_mm_examples_path = os.path.join(datasets_dir, 'test_mm_examples.pt')
    if os.path.exists(datasets_dir):
        logging.info(f'Loading saved dataset files from {datasets_dir}')
        train_examples = torch.load(train_examples_path)
        valid_m_examples = torch.load(valid_m_examples_path)
        valid_mm_examples = torch.load(valid_mm_examples_path)
        test_m_examples = torch.load(test_m_examples_path)
        test_mm_examples = torch.load(test_mm_examples_path)
        train_dataset = data.Dataset(
            examples=train_examples,
            fields=[('premise', text_field), ('hypothesis', text_field),
                    ('label', label_field), ('pair_id', pair_id_field)])
        train_dataset.sort_key = datasets.SNLI.sort_key
        valid_m_dataset = data.Dataset(
            examples=valid_m_examples,
            fields=[('premise', text_field), ('hypothesis', text_field),
                    ('label', label_field), ('pair_id', pair_id_field)])
        valid_m_dataset.sort_key = datasets.SNLI.sort_key
        valid_mm_dataset = data.Dataset(
            examples=valid_mm_examples,
            fields=[('premise', text_field), ('hypothesis', text_field),
                    ('label', label_field), ('pair_id', pair_id_field)])
        valid_mm_dataset.sort_key = datasets.SNLI.sort_key
        test_m_dataset = data.Dataset(
            examples=test_m_examples,
            fields=[('premise', text_field), ('hypothesis', text_field),
                    ('label', label_field), ('pair_id', pair_id_field)])
        test_m_dataset.sort_key = datasets.SNLI.sort_key
        test_mm_dataset = data.Dataset(
            examples=test_mm_examples,
            fields=[('premise', text_field), ('hypothesis', text_field),
                    ('label', label_field), ('pair_id', pair_id_field)])
        test_mm_dataset.sort_key = datasets.SNLI.sort_key
    else:
        train_examples = create_examples(
            path=os.path.join(mnli_dir, 'multinli_0.9_train.jsonl'),
            text_field=text_field, label_field=label_field, pair_id_field=pair_id_field)
        valid_m_examples = create_examples(
            path=os.path.join(mnli_dir, 'multinli_0.9_dev_matched.jsonl'),
            text_field=text_field, label_field=label_field, pair_id_field=pair_id_field)
        valid_mm_examples = create_examples(
            path=os.path.join(mnli_dir, 'multinli_0.9_dev_mismatched.jsonl'),
            text_field=text_field, label_field=label_field, pair_id_field=pair_id_field)
        test_m_examples = create_examples(
            path=os.path.join(mnli_dir, 'multinli_0.9_test_matched_unlabeled.jsonl'),
            text_field=text_field, label_field=label_field, pair_id_field=pair_id_field)
        test_mm_examples = create_examples(
            path=os.path.join(mnli_dir, 'multinli_0.9_test_mismatched_unlabeled.jsonl'),
            text_field=text_field, label_field=label_field, pair_id_field=pair_id_field)
        train_dataset = data.Dataset(
            examples=train_examples,
            fields=[('premise', text_field), ('hypothesis', text_field),
                    ('label', label_field), ('pair_id', pair_id_field)])
        train_dataset.sort_key = datasets.SNLI.sort_key
        valid_m_dataset = data.Dataset(
            examples=valid_m_examples,
            fields=[('premise', text_field), ('hypothesis', text_field),
                    ('label', label_field), ('pair_id', pair_id_field)])
        valid_m_dataset.sort_key = datasets.SNLI.sort_key
        valid_mm_dataset = data.Dataset(
            examples=valid_mm_examples,
            fields=[('premise', text_field), ('hypothesis', text_field),
                    ('label', label_field), ('pair_id', pair_id_field)])
        valid_mm_dataset.sort_key = datasets.SNLI.sort_key
        test_m_dataset = data.Dataset(
            examples=test_m_examples,
            fields=[('premise', text_field), ('hypothesis', text_field),
                    ('label', label_field), ('pair_id', pair_id_field)])
        test_m_dataset.sort_key = datasets.SNLI.sort_key
        test_mm_dataset = data.Dataset(
            examples=test_mm_examples,
            fields=[('premise', text_field), ('hypothesis', text_field),
                    ('label', label_field), ('pair_id', pair_id_field)])
        test_mm_dataset.sort_key = datasets.SNLI.sort_key
        os.makedirs(datasets_dir)
        torch.save(train_dataset.examples, train_examples_path)
        torch.save(valid_m_dataset.examples, valid_m_examples_path)
        torch.save(valid_mm_dataset.examples, valid_mm_examples_path)
        torch.save(test_m_dataset.examples, test_m_examples_path)
        torch.save(test_mm_dataset.examples, test_mm_examples_path)
    return (train_dataset, valid_m_dataset, valid_mm_dataset,
            test_m_dataset, test_mm_dataset)


def trim_dataset(dataset, max_length):
    for ex in dataset.examples:
        ex.premise = ex.premise[:max_length]
        ex.hypothesis = ex.hypothesis[:max_length]
