import argparse
import os

import torch
import yaml
from torchtext import data
from tqdm import tqdm

from .model import MNLIModel
from .data import load_data
from .data import TextField, LabelField


def generate(args):
    device = torch.device(args.device)

    text_field = TextField()
    label_field = LabelField()
    (train_dataset, valid_m_dataset, valid_mm_dataset,
     test_m_dataset, test_mm_dataset) = load_data(
        root='data', text_field=text_field, label_field=label_field)
    # Our model will be run in 'open-vocabulary' mode.
    text_field.build_vocab(train_dataset, valid_m_dataset, valid_mm_dataset,
                           test_m_dataset, test_mm_dataset)
    label_field.build_vocab(train_dataset)

    if args.test_type == 'matched':
        test_dataset = test_m_dataset
    elif args.test_type == 'mismatched':
        test_dataset = test_mm_dataset
    else:
        raise ValueError('Invalid valid type')

    test_loader = data.Iterator(
        dataset=test_dataset, batch_size=args.batch_size, train=False, device=device)

    config_path = os.path.join(os.path.dirname(args.model), 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    model = MNLIModel(num_words=len(text_field.vocab),
                      num_classes=len(label_field.vocab),
                      **config['model'])
    print(model)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_intrinsic_params = num_params - model.word_embedding.weight.numel()
    print(f'* # of params: {num_params}')
    print(f'  - Intrinsic: {num_intrinsic_params}')
    print(f'  - Word embedding: {num_params - num_intrinsic_params}')

    def run_iter(batch):
        pre_text, pre_length = batch.premise
        hyp_text, hyp_length = batch.hypothesis
        logit = model(pre_inputs=pre_text, pre_length=pre_length,
                      hyp_inputs=hyp_text, hyp_length=hyp_length)
        pred = logit.max(1)[1].cpu().tolist()
        return pred

    model.eval()
    torch.set_grad_enabled(False)

    examples_dict = {}
    for ex in test_dataset:
        examples_dict[ex.pair_id] = ex

    output_file = open(args.out, 'w')
    output_file.write('pairId,gold_label\n')
    for test_batch in tqdm(test_loader):
        pair_ids = test_batch.pair_id.cpu().tolist()
        labels = [label_field.vocab.itos[p] for p in run_iter(test_batch)]
        for pair_id, label in zip(pair_ids, labels):
            output_file.write(f'{pair_id},{label}\n')
    output_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-type', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
