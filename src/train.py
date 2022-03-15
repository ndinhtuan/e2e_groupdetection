from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from lib.models.model import create_group_model
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    if opt.num_workers > 0:
        torch.set_num_threads(opt.num_workers)
    
    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    valset_paths = data_config['val']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    train_dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    val_dataset = Dataset(opt, dataset_root, valset_paths, (1088, 608), augment=True, transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, train_dataset)
    print(opt)


    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    if len(opt.gpus) == 0:
        opt.device = torch.device('cpu')
    elif len(opt.gpus) == 1:
        opt.device = torch.device(f'cuda:{opt.gpus[0]}')
    else:
        opt.device = torch.device('cuda')

    print('Creating model...')
    print("DEVICES", os.environ['CUDA_VISIBLE_DEVICES'], opt.gpus_str, opt.gpus)

    model = create_model(opt.arch, opt.heads, opt.head_conv)
    group_model = create_group_model(opt)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    # Get dataloader
    print("Num workers", opt.num_workers)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    print('Starting training...')
    Trainer = train_factory[opt.task]
    logger = Logger(opt)
    dict_model = {}
    dict_model["main_model"] = model
    dict_model["group_model"] = group_model
    trainer = Trainer(opt, dict_model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    # Load detection model from pretrain path
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

    # Load group model from pretrain path
    if opt.load_model_group != '':
        group_model = load_model(group_model, opt.load_model_group)

    # for param in model.parameters():
    #     param.requires_grad = False

    best_val_loss = 99999999999
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        print("Epoch: ", log_dict_train)
        logger.write('epoch: {} |'.format(epoch))

        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'val_model_{}.pth'.format(mark)),
                    epoch, model, optimizer)
            save_model(os.path.join(opt.save_dir, 'val_group_model_{}.pth'.format(mark)),
                    epoch, group_model, optimizer)
        
            
        logger.write('\n')
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                   epoch, model, optimizer)
        save_model(os.path.join(opt.save_dir, 'group_model_{}.pth'.format(epoch)),
                   epoch, group_model, optimizer)
        
        ### Evaluate on val set###
        if opt.val_intervals == 0 or epoch % opt.val_intervals == 0:
            print("Evaluating on validation set")
            log_dict_val, _ = trainer.val(epoch, val_loader)
            for k, v in log_dict_train.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            val_loss = log_dict_val["loss"]
            print(f"Val loss: {val_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                            epoch, model, optimizer)
                save_model(os.path.join(opt.save_dir, 'group_model_best.pth'),
                            epoch, group_model, optimizer)
                print(f"Saved best model {epoch}. Loss={best_val_loss}")
            
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
