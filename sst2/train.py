import argparse
import logging
import os
import pprint
import random

import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim import lr_scheduler
from torchtext import data

from .model import SSTModel
from .data import load_data
from .data import TextField, LabelField


log_formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_log_handler = logging.StreamHandler()
console_log_handler.setFormatter(log_formatter)
logger.addHandler(console_log_handler)


def train(args):
    device = torch.device(args.device)

    text_field = TextField()
    label_field = LabelField()
    train_dataset, valid_dataset, test_dataset = load_data(
        root='data', text_field=text_field, label_field=label_field)
    # Our model will be run in 'open-vocabulary' mode.
    text_field.build_vocab(train_dataset, valid_dataset, test_dataset)
    label_field.build_vocab(train_dataset)
    text_field.vocab.load_vectors(args.word_vector)

    train_loader, valid_loader, test_loader = data.Iterator.splits(
        datasets=(train_dataset, valid_dataset, test_dataset),
        batch_size=args.batch_size, device=device)

    config_path = os.path.join(args.save_dir, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    model = SSTModel(num_words=len(text_field.vocab),
                     num_classes=len(label_field.vocab),
                     **config['model'])
    model.word_embedding.weight.data.set_(text_field.vocab.vectors)
    model.word_embedding.weight.requires_grad = not args.fix_word_embeddings
    print(model)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_intrinsic_params = num_params - model.word_embedding.weight.numel()
    logger.info(f'* # of params: {num_params}')
    logger.info(f'  - Intrinsic: {num_intrinsic_params}')
    logger.info(f'  - Word embedding: {num_params - num_intrinsic_params}')

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'adam':
        optimizer = optim.Adam(trainable_params)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(trainable_params)
    assert not args.warm_restart or args.cosine_lr
    if args.cosine_lr:
        if not args.warm_restart:
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=len(train_loader) * args.max_epoch)
        else:
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=len(train_loader) * 2)
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='max', factor=0.5, patience=4, verbose=True)
    criterion = nn.CrossEntropyLoss()

    def run_iter(batch):
        text, length = batch.text
        label = batch.label
        logit = model(inputs=text, length=length)
        clf_loss = criterion(input=logit, target=label)
        pred = logit.max(1)[1]
        accuracy = torch.eq(pred, label).float().mean()
        if model.training:
            if args.l2_weight > 0:
                l2_norm = sum(p.pow(2).sum() for p in trainable_params).sqrt()
            else:
                l2_norm = 0
            loss = clf_loss + args.l2_weight*l2_norm
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=5)
            optimizer.step()
        return clf_loss.item(), accuracy.item()

    def validate(loader):
        model.eval()
        clf_loss_sum = accuracy_sum = 0
        num_valid_data = len(loader.dataset)
        with torch.no_grad():
            for valid_batch in loader:
                clf_loss, accuracy = run_iter(valid_batch)
                clf_loss_sum += clf_loss * valid_batch.batch_size
                accuracy_sum += accuracy * valid_batch.batch_size
        clf_loss = clf_loss_sum / num_valid_data
        accuracy = accuracy_sum / num_valid_data
        return clf_loss, accuracy

    train_summary_writer = SummaryWriter(
        os.path.join(args.save_dir, 'log', 'train'))
    valid_summary_writer = SummaryWriter(
        os.path.join(args.save_dir, 'log', 'valid'))

    validate_every = len(train_loader) // args.verbosity
    best_valid_accuracy = 0
    global_step = 0
    logger.info('Training starts!')
    for train_batch in train_loader:
        if not model.training:
            model.train()
        train_clf_loss, train_accuracy = run_iter(train_batch)
        global_step += 1
        if args.cosine_lr:
            if not args.warm_restart:
                scheduler.step()
            else:
                if scheduler.last_epoch == scheduler.T_max:
                    scheduler.T_max = scheduler.T_max * 2
                    scheduler.step(0)
                    logger.info('Warm-restarted the learning rate!')
                else:
                    scheduler.step()

        train_summary_writer.add_scalar(
            tag='clf_loss', scalar_value=train_clf_loss,
            global_step=global_step)
        train_summary_writer.add_scalar(
            tag='accuracy', scalar_value=train_accuracy,
            global_step=global_step)

        if global_step % validate_every == 0:
            progress = train_loader.iterations / len(train_loader)
            logger.info(f'* Epoch {progress:.2f}')
            logger.info(f'  - lr = {optimizer.param_groups[0]["lr"]:.6f}')
            logger.info(f'  - Validation starts')
            valid_clf_loss, valid_accuracy = validate(valid_loader)
            _, test_accuracy = validate(test_loader)
            if not args.cosine_lr:
                scheduler.step(valid_accuracy)
            valid_summary_writer.add_scalar(
                tag='clf_loss', scalar_value=valid_clf_loss,
                global_step=global_step)
            valid_summary_writer.add_scalar(
                tag='accuracy', scalar_value=valid_accuracy,
                global_step=global_step)
            valid_summary_writer.add_scalar(
                tag='lr', scalar_value=optimizer.param_groups[0]['lr'],
                global_step=global_step)
            logger.info(f'  - Valid clf loss: {valid_clf_loss:.5f}')
            logger.info(f'  - Valid accuracy: {valid_accuracy:.5f}')
            logger.info(f'  - Test accuracy: {test_accuracy:.5f}')
            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                model_filename = (f'best-{progress:.2f}'
                                  f'-{valid_clf_loss:.5f}'
                                  f'-{valid_accuracy:.5f}.pt')
                model_path = os.path.join(args.save_dir, model_filename)
                torch.save(model.state_dict(), model_path)
                logger.info(f'  - Saved the new best model to: {model_path}')
            elif args.save_every_epoch and global_step % (validate_every * 10) == 0:
                model_filename = (f'model-{progress:.2f}'
                                  f'-{valid_clf_loss:.5f}'
                                  f'-{valid_accuracy:.5f}.pt')
                model_path = os.path.join(args.save_dir, model_filename)
                torch.save(model.state_dict(), model_path)
                logger.info(f'  - Saved the new model to: {model_path}')

        if train_loader.epoch > args.max_epoch:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word-dim', type=int, default=300)
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--mlp-hidden-dim', type=int, default=300)
    parser.add_argument('--mlp-num-layers', type=int, default=1)
    parser.add_argument('--mlp-bn', default=False, action='store_true')
    parser.add_argument('--emb-dropout', type=float, default=0.5)
    parser.add_argument('--enc-bidir', default=False, action='store_true')
    parser.add_argument('--enc-bidir-init', default=False, action='store_true')
    parser.add_argument('--enc-lstm-type', default='ca')
    parser.add_argument('--enc-num-layers', type=int, default=2)
    parser.add_argument('--enc-pool-type', default='max')
    parser.add_argument('--fuse-type', default='add')
    parser.add_argument('--enc-dropout', type=float, default=0)
    parser.add_argument('--clf-dropout', type=float, default=0.5)
    parser.add_argument('--word-vector', default='glove.840B.300d')
    parser.add_argument('--fix-word-embeddings', default=False,
                        action='store_true')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--verbosity', type=int, default=2)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument('--l2-weight', type=float, default=0)
    parser.add_argument('--cosine-lr', default=False, action='store_true')
    parser.add_argument('--warm-restart', default=False, action='store_true')
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--save-every-epoch', default=False, action='store_true')
    parser.add_argument('--optimizer', default='adam')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = {'model': {'word_dim': args.word_dim,
                        'hidden_dim': args.hidden_dim,
                        'mlp_hidden_dim': args.mlp_hidden_dim,
                        'mlp_num_layers': args.mlp_num_layers,
                        'mlp_use_bn': args.mlp_bn,
                        'enc_lstm_type': args.enc_lstm_type,
                        'enc_bidir': args.enc_bidir,
                        'enc_bidir_init': args.enc_bidir_init,
                        'enc_num_layers': args.enc_num_layers,
                        'enc_pool_type': args.enc_pool_type,
                        'fuse_type': args.fuse_type,
                        'emb_dropout_prob': args.emb_dropout,
                        'enc_dropout_prob': args.enc_dropout,
                        'clf_dropout_prob': args.clf_dropout},
              'train': {'batch_size': args.batch_size,
                        'verbosity': args.verbosity,
                        'word_vector': args.word_vector,
                        'fix_word_embeddings': args.fix_word_embeddings,
                        'l2_weight': args.l2_weight,
                        'cosine_lr': args.cosine_lr,
                        'warm_restart': args.warm_restart,
                        'seed': args.seed}
              }
    pprint.pprint(config)
    os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    file_log_handler = logging.FileHandler(
        os.path.join(args.save_dir, 'train.log'))
    file_log_handler.setFormatter(log_formatter)
    logger.addHandler(file_log_handler)

    train(args)


if __name__ == '__main__':
    main()
