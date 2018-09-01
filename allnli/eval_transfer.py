import argparse
import os
import random
import time

import yaml

import torch
from torch import nn

import senteval
from .model import AllNLIModel
from .data import TextField, LabelField


torch.manual_seed(123)
random.seed(123)


def prepare(params, samples):
    text_field = params.fields['text']
    print('Building the new vocabulary')
    text_field.build_vocab(samples)
    text_field.vocab.load_vectors('glove.840B.300d')
    word_embedding = nn.Embedding(num_embeddings=len(text_field.vocab),
                                  embedding_dim=300)
    nn.init.normal_(word_embedding.weight, mean=0, std=0.5)
    print(f'New word embedding: {word_embedding.weight.shape}')
    word_embedding.weight.data.set_(text_field.vocab.vectors)
    params.model.word_embedding = word_embedding
    params.model.to(params.device)
    params.model.eval()


def batcher(params, batch):
    tic = time.time()
    text_field = params.fields['text']
    inputs, length = text_field.process(batch, device=params.device)
    with torch.no_grad():
        sentence_vector = params.model.encode(inputs=inputs, length=length)
    sentence_vector = sentence_vector.cpu().numpy()
    speed = len(sentence_vector) / (time.time() - tic)
    print(f'Speed: {speed:.1f} sentences/s (bsize={params.batch_size})')
    return sentence_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--prototyping', default=False, action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(args.model), 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    text_field = TextField()
    text_field.tokenize = lambda s: s.split()
    label_field = LabelField()

    model_weights = torch.load(args.model, map_location='cpu')
    num_words = model_weights['word_embedding.weight'].shape[0]
    num_classes = model_weights['output_linear.bias'].shape[0]
    model = AllNLIModel(num_words=num_words, num_classes=num_classes,
                        **config['model'])
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.to(args.device)
    print(f'Loaded the model from {args.model}')

    params_senteval = {'task_path': args.data, 'usepytorch': True}
    if args.prototyping:
        params_senteval['kfold'] = 5
        params_senteval['classifier'] = {
            'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3,
            'epoch_size': 2
        }
    else:
        params_senteval['kfold'] = 10
        params_senteval['classifier'] = {
            'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5,
            'epoch_size': 4}

    params_senteval['config'] = config
    params_senteval['fields'] = {'text': text_field, 'label': label_field}
    params_senteval['device'] = args.device
    params_senteval['model'] = model

    se = senteval.engine.SE(params_senteval, batcher=batcher, prepare=prepare)
    # transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5' 'TREC', 'MRPC',
    #                   'SICKRelatedness', 'SICKEntailment', 'STS14']
    transfer_tasks = ['MR']
    results = se.eval(transfer_tasks)
    print(results)


if __name__ == '__main__':
    main()
