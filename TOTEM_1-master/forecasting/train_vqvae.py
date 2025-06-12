import argparse
import comet_ml
import json
import numpy as np
import os
import pdb
import random
import time
import torch
from lib.models import get_model_class
from time import gmtime, strftime


def main(device, config, save_dir, logger, data_init_loc, args):
    # Create/overwrite checkpoints folder and results folder
    if os.path.exists(os.path.join(save_dir, 'checkpoints')):
        print('Checkpoint Directory Already Exists - if continue will overwrite files inside. Press c to continue.')
        pdb.set_trace()
    else:
        os.makedirs(os.path.join(save_dir, 'checkpoints'))

    if logger is not None:
        logger.log_parameters(config)

    # Run start training
    vqvae_config, summary = start_training(device=device,
                                           vqvae_config=config['vqvae_config'],
                                           save_dir=save_dir,
                                           logger=logger,
                                           data_init_loc=data_init_loc,
                                           args=args)

    # Save config file
    config['vqvae_config'] = vqvae_config
    print('CONFIG FILE TO SAVE:', config)

    if os.path.exists(os.path.join(save_dir, 'configs')):
        print('Saved Config Directory Already Exists - if continue will overwrite files inside. Press c to continue.')
        pdb.set_trace()
    else:
        os.makedirs(os.path.join(save_dir, 'configs'))

    with open(os.path.join(save_dir, 'configs', 'config_file.json'), 'w+') as f:
        json.dump(config, f, indent=4)

    summary['log_path'] = os.path.join(save_dir)
    master['summaries'] = summary
    print('MASTER FILE:', master)
    with open(os.path.join(save_dir, 'master.json'), 'w') as f:
        json.dump(master, f, indent=4)


def start_training(device, vqvae_config, save_dir, logger, data_init_loc, args):
    summary = {}

    if 'general_seed' not in vqvae_config:
        vqvae_config['general_seed'] = random.randint(0, 9999)

    general_seed = vqvae_config['general_seed']
    summary['general_seed'] = general_seed
    torch.manual_seed(general_seed)
    random.seed(general_seed)
    np.random.seed(general_seed)
    torch.backends.cudnn.deterministic = True

    summary['data initialization location'] = data_init_loc
    summary['device'] = device

    model_class = get_model_class(vqvae_config['model_name'].lower())
    model = model_class(vqvae_config)

    print('Total # trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if vqvae_config['pretrained']:
        model = torch.load(vqvae_config['pretrained'])
    summary['vqvae_config'] = vqvae_config

    start_time = time.time()
    model = train_model(model, device, vqvae_config, save_dir, logger, args=args)
    torch.save(model, os.path.join(save_dir, 'checkpoints/final_model.pth'))

    summary['total_time'] = round(time.time() - start_time, 3)
    return vqvae_config, summary


def train_model(model, device, vqvae_config, save_dir, logger, args):
    optimizer = model.configure_optimizers(lr=vqvae_config['learning_rate'])

    model.to(device)
    start_time = time.time()

    print('BATCHSIZE:', args.batchsize)
    train_loader, vali_loader, test_loader = create_datloaders(batchsize=args.batchsize,
                                                               dataset=vqvae_config["dataset"],
                                                               base_path=args.base_path)

    for epoch in range(int((vqvae_config['num_training_updates'] / len(train_loader)) + 0.5)):
        model.train()
        for i, (batch_x) in enumerate(train_loader):
            tensor_all_data_in_batch = torch.tensor(batch_x, dtype=torch.float, device=device)

            loss, vq_loss, recon_error, x_recon, perplexity, embedding_weight, encoding_indices, encodings = \
                model.shared_eval(tensor_all_data_in_batch, optimizer, 'train', comet_logger=logger)

        if epoch % 10000 == 0:
            torch.save(model, os.path.join(save_dir, f'checkpoints/model_epoch_{epoch}.pth'))
            print('Saved model from epoch ', epoch)

    print('total time: ', round(time.time() - start_time, 3))
    return model


def create_datloaders(batchsize=100, dataset="dummy", base_path='dummy'):
    dataset_paths = {
        'weather': 'weather',
        'electricity': 'electricity',
        'traffic': 'traffic',
        'ETTh1': 'ETTh1',
        'ETTm1': 'ETTm1',
        'ETTh2': 'ETTh2',
        'ETTm2': 'ETTm2',
        'all': 'all'
    }

    if dataset in dataset_paths:
        print(dataset)
        full_path = os.path.join(base_path, dataset_paths[dataset])
    else:
        print('Not done yet')
        pdb.set_trace()

    train_data = np.load(os.path.join(full_path, "train_data_x.npy"), allow_pickle=True)
    val_data = np.load(os.path.join(full_path, "val_data_x.npy"), allow_pickle=True)
    test_data = np.load(os.path.join(full_path, "test_data_x.npy"), allow_pickle=True)

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batchsize,
                                                   shuffle=True,
                                                   num_workers=10,
                                                   drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=batchsize,
                                                 shuffle=False,
                                                 num_workers=10,
                                                 drop_last=False)

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=batchsize,
                                                  shuffle=False,
                                                  num_workers=10,
                                                  drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        required=True,
                        help='path to specific config file once already in the config folder')
    parser.add_argument('--model_init_num_gpus', type=int,
                        required=False, default=0,
                        help='number of gpus to use, 0 indexed, so if you want 1 gpu say 0')
    parser.add_argument('--data_init_cpu_or_gpu', type=str,
                        required=False,
                        help='the data initialization location')
    parser.add_argument('--comet_log', action='store_true',
                        required=False,
                        help='whether to log to comet online')
    parser.add_argument('--comet_tag', type=str,
                        required=False,
                        help='the experimental tag to add to comet - this should be the person running the exp')
    parser.add_argument('--comet_name', type=str,
                        required=False,
                        help='the experiment name to add to comet')
    parser.add_argument('--save_path', type=str,
                        required=True,
                        help='where were going to save the checkpoints')
    parser.add_argument('--base_path', type=str,
                        default='',
                        help='saved revin data to train model')
    parser.add_argument('--batchsize', type=int,
                        required=True,
                        help='batchsize')

    args = parser.parse_args()

    # Load JSON config file
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    print('Config folder:\t {}'.format(args.config_path))
    print('Running Config:', args.config_path)

    save_folder_name = 'CD' + str(config['vqvae_config']['embedding_dim']) + '_CW' + str(
        config['vqvae_config']['num_embeddings']) + '_CF' + str(
        config['vqvae_config']['compression_factor']) + '_BS' + str(args.batchsize) + '_ITR' + str(
        config['vqvae_config']['num_training_updates'])

    save_dir = os.path.join(args.save_path, save_folder_name)

    master = {
        'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'config file': args.config_path,
        'save directory': save_dir,
        'gpus': args.model_init_num_gpus,
    }

    if args.comet_log:
        comet_logger = comet_ml.Experiment(
            api_key=config['comet_config']['api_key'],
            project_name=config['comet_config']['project_name'],
            workspace=config['comet_config']['workspace'],
        )
        if args.comet_tag:
            comet_logger.add_tag(args.comet_tag)
        if args.comet_name:
            comet_logger.set_name(args.comet_name)
    else:
        print('PROBLEM: not saving to comet')
        comet_logger = None
        pdb.set_trace()

    if torch.cuda.is_available() and args.model_init_num_gpus >= 0:
        assert args.model_init_num_gpus < torch.cuda.device_count()
        device = 'cuda:{:d}'.format(args.model_init_num_gpus)
    else:
        device = 'cpu'

    if args.data_init_cpu_or_gpu == 'gpu':
        data_init_loc = device
    else:
        data_init_loc = 'cpu'

    main(device, config, save_dir, comet_logger, data_init_loc, args)
