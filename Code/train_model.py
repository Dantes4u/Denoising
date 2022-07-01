import os
import numpy as np
import yaml
import random
import json
import torch
import torch.nn as nn
from tqdm import tqdm

import models

from logs import Logger
from experiment import Experiment, SplitExperiment
from data import DataHolder, DataIter
from ranger21 import Ranger21
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer:
    def __init__(self, logs_dir, config_name, gpu):
        self.gpu = gpu
        self.config = self._load_config(config_name)
        self.logs_dir = logs_dir
        if self.gpu == 0:
            self.logger = Logger()

        self.params = self.config['Parameters']
        self.data_config = self.config['Data']
        self.model_config = self.config['Model']
        self.transform_config = self.config['Transform']
        self.experiment_config = self.config['Experiment']

        self.save_all = self.data_config['save_all']
        self.distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
        if self.distributed:
            torch.cuda.set_device(gpu)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.num_gpus = torch.distributed.get_world_size()
        else:
            self.num_gpus = 1

        self.data_holder = DataHolder(self.data_config)
        self.num_epochs = self.params['num_epochs']
        self.batch_size = self.params['batch_size']

        self._seed()
        self._init_model()
        self._init_loaders()
        self._init_optimizer()
        self._init_scheduler()
        self._init_loss()
        self._init_experiment()

    def _seed(self):
        seed_state = {}
        for name in ('random', 'torch', 'numpy'):
            seed_state[name] = np.random.randint(0, 900)
            if self.distributed:
                seed_state[name] += torch.distributed.get_rank()

        random.seed(seed_state['random'])
        torch.manual_seed(seed_state['torch'])
        torch.cuda.manual_seed(seed_state['torch'])
        np.random.seed(seed=seed_state['numpy'])

        if self.gpu == 0:
            with open(os.path.join(self.logs_dir, 'seed_state.json'), 'w+') as output_file:
                json.dump(seed_state, output_file)
            
    def _init_model(self):

        model_name = self.model_config['name']
        self.model = models.MODELS[model_name]()

        if 'params' in self.model_config:
            self.checkpoint = torch.load(self.model_config['params'])
            self.model.load_state_dict(self.checkpoint['model_state_dict'])

        self.model = self.model.cuda()
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.gpu], output_device=self.gpu)
        # self.model = nn.DataParallel(self.model).cuda()

    def _init_optimizer(self):
        initial_lr = float(self.params['initial_lr'])
        wd = float(self.params['weight_decay'])
        lookahead_mergetime = int(self.params['lookahead_mergetime'])
        #self.optimizer = Ranger21(
        #   self.model.parameters(),
        #   lookahead_mergetime=lookahead_mergetime,
        #   lr=initial_lr,
        #   weight_decay=wd,
        #   num_epochs=self.num_epochs,
        #   num_batches_per_epoch=self.train_iters
        #)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                            lr=initial_lr,
                                            weight_decay=wd,
                                            )
        if 'params' in self.model_config:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            for g in self.optimizer.param_groups:
                g['lr'] = initial_lr
                g['initial_lr'] = initial_lr
        self.scaler = torch.cuda.amp.GradScaler()

    def _init_scheduler(self):
        train_iters = len(self.data_holder.datasets['train']) // self.batch_size + 1
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            train_iters * self.num_epochs,
        )

    def _init_loss(self):
        self.mse = nn.MSELoss(reduction='mean')

    def _init_loaders(self):
        if self.gpu == 0:
            self.logger.info("Building random loader for train data")
        train_iter = DataIter(self.data_holder, self.transform_config, 'train', gpu=self.gpu)
        self.sampler = torch.utils.data.distributed.DistributedSampler(train_iter) if self.distributed else None
        self.train_loader = torch.utils.data.DataLoader(
            train_iter,
            sampler=self.sampler,
            batch_size=self.batch_size,
            num_workers=self.params['num_workers'],
            shuffle=(self.sampler is None),
            pin_memory=self.params['pin_memory'],
            prefetch_factor=self.params['prefetch_factor']
        )
        self.train_iters = len(self.train_loader)

        self.test_loaders = {}
        for test_name in self.data_holder.datasets:
            if test_name == 'train':
                continue
            if self.gpu == 0:
                self.logger.info("Building loader for test data: \"{}\".".format(test_name))
            test_iter = DataIter(self.data_holder, self.transform_config, test_name, gpu=0)
            test_loader = torch.utils.data.DataLoader(
                test_iter,
                batch_size=self.batch_size,
                num_workers=self.params['num_workers'],
                shuffle=False,
                pin_memory=self.params['pin_memory'],
                prefetch_factor=self.params['prefetch_factor']
            )
            self.test_loaders[test_name] = {
                'loader': test_loader,
                'iters': len(test_loader)
            }

    def _init_experiment(self):
        self.experiment = Experiment(self.data_holder.domains, self.experiment_config, self.logs_dir)

    def train(self, epoch):
        global split_experiment
        if self.distributed:
            self.sampler.set_epoch(epoch)
        self.model.train()
        if self.gpu == 0:
            self.logger.info('Training epoch {}/{}.'.format(epoch, self.num_epochs))
            self.logger.info('Learning rate schedule: {}'.format(self.lr_scheduler.get_last_lr()[0]))

        if self.gpu == 0:
            self.experiment.setup(epoch)
            split_experiment = SplitExperiment(domains=self.data_holder.domains, split_name='Train')
        iterator = self.train_loader
        if self.gpu == 0:
            iterator = tqdm(iterator, total=self.train_iters, unit='batch')
        for noisy, clean in iterator:
            noisy = noisy.cuda(self.gpu)
            clean = clean.cuda(self.gpu)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = self.model(noisy)
                mse = self.mse(pred, clean)
                loss = mse
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
            if self.gpu == 0:
                loss = loss.cpu().detach().numpy().astype(np.float32)
                clean = clean.cpu().detach().numpy()
                split_experiment.update(clean, loss)
                iterator.set_description('loss: {:.3f}, lr: {:.10f}'.format(loss, self.lr_scheduler.get_last_lr()[0]))
        if self.gpu == 0:
            self.experiment.update_split(split_experiment.get_split())

    def test(self, epoch):
        self.model.eval()
        for test_name in self.test_loaders:
            self.logger.info('Validating epoch {} at \'{}\'.'.format(epoch, test_name))
            loader = self.test_loaders[test_name]['loader']
            iters = self.test_loaders[test_name]['iters']
            split_experiment = SplitExperiment(domains=self.data_holder.domains, split_name=test_name)
            iterator = tqdm(loader, total=iters, unit='batch')
            for noisy, clean in iterator:
                noisy = noisy.cuda(self.gpu)
                clean = clean.cuda(self.gpu)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        pred = self.model(noisy)
                        mse = self.mse(pred, clean)
                        loss = mse
                    loss = loss.cpu().detach().numpy().astype(np.float32)
                    clean = clean.cpu().detach().numpy()
                    split_experiment.update(clean, loss)
                    iterator.set_description('loss: {:.3f}'.format(loss))
            self.experiment.update_split(split_experiment.get_split())

    def render(self):
        self.experiment.render()

    def export_model(self, epoch):
        model_name = self.model_config['name'].lower()
        model_output_dir = os.path.join(self.logs_dir, 'Models', '{:03d}'.format(epoch))
        os.makedirs(model_output_dir, exist_ok=True)

        model_output_path = os.path.join(model_output_dir, f'{model_name}_{epoch}.pt')

        model_state_dict = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        if self.save_all == True:
            optimizer_state_dict = self.optimizer.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict}, model_output_path)
        else:
            torch.save(model_state_dict, model_output_path)

    @staticmethod
    def _load_config(config_name):
        config_path = os.path.join('config', 'base', config_name)
        with open(config_path, 'r') as input_file:
            config = yaml.safe_load(input_file)
        return config


def train_model(logs_dir, config_name, gpu):
    trainer = Trainer(logs_dir, config_name, gpu)
    for epoch in range(trainer.num_epochs):
        trainer.train(epoch)
        if gpu == 0:
            trainer.test(epoch)
            trainer.render()
            trainer.export_model(epoch)
