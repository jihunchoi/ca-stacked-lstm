import argparse
import os

import torch
import yaml
from torchtext import data
from tqdm import tqdm

from .model import SNLIModel
from .data import load_data
from .data import TextField, LabelField


def test(args):
    device = torch.device(args.device)

    text_field = TextField()
    label_field = LabelField()
    train_dataset, valid_dataset, test_dataset = load_data(
        root='data', text_field=text_field, label_field=label_field)
    # Our model will be run in 'open-vocabulary' mode.
    text_field.build_vocab(train_dataset, valid_dataset, test_dataset)
    label_field.build_vocab(train_dataset)

    test_loader = data.Iterator(
        dataset=test_dataset, batch_size=args.batch_size, train=False,
        device=device)

    config_path = os.path.join(os.path.dirname(args.model), 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    model = SNLIModel(num_words=len(text_field.vocab),
                      num_classes=len(label_field.vocab),
                      **config['model'])
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_intrinsic_params = num_params - model.word_embedding.weight.numel()
    print(f'* # of params: {num_params}')
    print(f'  - Intrinsic: {num_intrinsic_params}')
    print(f'  - Word embedding: {num_params - num_intrinsic_params}')

    model.eval()
    num_correct = 0
    with torch.no_grad():
        for test_batch in tqdm(test_loader):
            pre_text, pre_length = test_batch.premise
            hyp_text, hyp_length = test_batch.hypothesis
            label = test_batch.label
            logit = model(pre_inputs=pre_text, pre_length=pre_length,
                          hyp_inputs=hyp_text, hyp_length=hyp_length)
            pred = logit.max(1)[1]
            num_correct_batch = torch.eq(label, pred).sum().item()
            num_correct += num_correct_batch
    print(f'# correct: {num_correct}')
    print(f'# total: {len(test_dataset)}')
    print(f'Accuracy: {num_correct / len(test_dataset):.5f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', default=128, type=int)
    args = parser.parse_args()

    test(args)


if __name__ == '__main__':
    main()
