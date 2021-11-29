from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import os
import logging
from logging import Logger
import yaml
import json
import time
from datetime import timedelta
import copy

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch import Tensor
import torch.nn as nn

class BaseModel(ABC):
    def __init__(self, logger: Logger = None, print_progress: bool = True, device: str = 'cuda:0', **kwargs):
        """

        """
        # where to print info
        self.print_fn = logger.info if logger else print

        self.device = device
        self.print_progress = print_progress

        self.outputs = {} # placeholder for any output to be saved in YAML
        self.extra_checkpoint_keys = [] # list of attribute name to save in checkpoint

    def save_outputs(self, export_path: str):
        """
        Save the output attribute dictionnary as a YAML or JSON specified by export_path.
        """
        if os.path.splitext(export_path)[-1] in ['.yml', '.yaml']:
            with open(export_path, "w") as f:
                yaml.dump(self.outputs, f)
        elif os.path.splitext(export_path)[-1] == '.json':
            with open(export_path, "w") as f:
                json.dump(self.outputs, f)

    @staticmethod
    def print_progessbar(n: int, max: int, name: str = '', size: int = 10, end_char: str = '', erase: bool = False):
        """
        Print a progress bar. To be used in a for-loop and called at each iteration
        with the iteration number and the max number of iteration.
        ------------
        INPUT
            |---- n (int) the iteration current number
            |---- max (int) the total number of iteration
            |---- name (str) an optional name for the progress bar
            |---- size (int) the size of the progress bar
            |---- end_char (str) the print end parameter to used in the end of the
            |                    progress bar (default is '')
            |---- erase (bool) whether to erase the progress bar when 100% is reached.
        OUTPUT
            |---- None
        """
        frmt = f"0{len(str(max))}d"
        print(f'{name} {n+1:{frmt}}/{max:{frmt}}'.ljust(len(name) + 12) \
            + f'|{"█"*int(size*(n+1)/max)}'.ljust(size+1) + f'| {(n+1)/max:.1%}'.ljust(6), \
            end='\r')

        if n+1 == max:
            if erase:
                print(' '.ljust(len(name) + size + 40), end='\r')
            else:
                print('')

class BaseModelSingle(BaseModel):
    def __init__(self, net: nn.Module, opt: Optimizer = None, sched: _LRScheduler = None,
                 logger: Logger = None, print_progress: bool = True, device: str = 'cuda:0', **kwargs):
        """
        Abtract class defining a moodel based on Pytorch. It allows to save/load the model and train/evaluate it.
        Classes inheriting from the BaseModel needs to be initialized with a nn.Modules. This network can be trained using
        the passed optimizer/lr_scheduler with the self.train() methods. To be used, the children class must define two
        abstract methods:
            1. `forward_loss(data: Tuple[Tensor])` : define the processing of 1 batch provided by the DataLoader. `data`
               is the tuple of tensors given by the DataLoader. This method should thus define how the data is i) unpacked
               ii) how the forward pass with self.net is done iii) and how the loss is computed. The method should then
               return the loss.
            2. `validate(loader: DataLoader)` : define how the model is validated at each epoch. It takes a DataLoader
               for the validation data as input and should return a dictionnary of properties to print in the epoch
               summary (as {property_name : str_property_value}). No validation is performed if no valid_loader is passed
               to self.train()

        Note: the BaseModel has a dictionnary as attributes (self.outputs) that allow to store some values (training time,
              validation scores, epoch evolution, etc). This dictionnary can be saved as a YAML file using the save_outputs
              method. Any other values can be added to the self.outputs using self.outputs["key"] = value.

              If Logger is None, the outputs are displayed using `print`.
        """
        super().__init__(logger=logger, print_progress=print_progress, device=device, **kwargs)

        self.net = net
        self.net = self.net.to(device)
        self.best_net = net
        self.best_metric = None
        self.optimizer = opt
        self.lr_scheduler = sched
        self.logger = logger

    def train(self, n_epochs: int, train_loader: DataLoader, valid_loader: DataLoader = None,
              extra_valid_args: List = [], extra_valid_kwargs: Dict = dict(),
              checkpoint_path: str = None, checkpoint_freq: int = 10,
              save_best_key : str = None, minimize_metric : bool = True, min_epoch_best : int = 0):
        """
        Train the self.net using the optimizer and scheduler using the data provided by the train_loader. At each epoch,
        the model can be validated using the valid_loader (if a valid loader is provided, the method self.validate must
        be implemented in the children). The model and training state is loaded/saved in a .pt file if checkpoint_path
        is provided. The model is then saved every checkpoint_freq epoch.

        The best model can be saved over the training processed based on one of the validation metric provided by the
        self.validate output dictionnary. The metric to use is specified by the string `save_best_key` and the argument
        `minimize_metric` define whether the metric must be minimized or maximized. A mininumm number of epoch to be
        performed before selcting the best model can be specified with 'min_epoch_best'.
        """
        assert self.optimizer is not None, "An optimizer must be provided to train the model."

        # Load checkpoint if any
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                n_epoch_finished = checkpoint['n_epoch_finished']
                self.net.load_state_dict(checkpoint['net_state'])
                self.net = self.net.to(self.device)
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                if save_best_key:
                    best_metric = checkpoint['best_metric']
                    best_epoch = checkpoint['best_epoch']
                    self.best_net.load_state_dict(checkpoint['best_net_state'])
                    self.best_net = self.best_net.to(self.device)

                if self.lr_scheduler:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_state'])

                epoch_loss_list = checkpoint['loss_evolution']

                for k in self.extra_checkpoint_keys:
                    setattr(self, k, checkpoint[k])

                self.print_fn(f'Resuming from Checkpoint with {n_epoch_finished} epoch finished.')
            except FileNotFoundError:
                self.print_fn('No Checkpoint found. Training from beginning.')
                n_epoch_finished = 0
                epoch_loss_list = [] # Placeholder for epoch evolution
        else:
            self.print_fn('No Checkpoint used. Training from beginning.')
            n_epoch_finished = 0
            epoch_loss_list = []

        self.net = self.net.to(self.device)
        # Train Loop
        for epoch in range(n_epoch_finished, n_epochs):
            self.net.train()
            epoch_start_time = time.time()
            epoch_loss = 0.0
            n_batches = len(train_loader)

            for b, data in enumerate(train_loader):
                # Gradient descent step
                self.optimizer.zero_grad()
                loss = self.forward_loss(data)
                # recover returned loss(es) value(s)
                if isinstance(loss, tuple):
                    loss, all_losses = loss
                    if b == 0:
                        train_losses = {name : 0.0 for name in all_losses.keys()}
                    train_losses = {name : (value + all_losses[name].item() if isinstance(all_losses[name], torch.Tensor) else value + all_losses[name]) for name, value in train_losses.items()}
                else:
                    if b == 0:
                        train_losses = {'Loss' : 0.0}
                    train_losses["Loss"] += loss.item()

                loss.backward()
                self.optimizer.step()

                if self.print_progress:
                    self.print_progessbar(b, n_batches, name='Train Batch', size=100, erase=True)

            # validate epoch
            if valid_loader:
                self.net.eval()
                valid_outputs = self.validate(valid_loader, epoch, *extra_valid_args, **extra_valid_kwargs)
                # unpack format dictionnary if provided
                if isinstance(valid_outputs, dict):
                    valid_outputs_format = {key : '' for key in valid_outputs.keys()}
                elif isinstance(valid_outputs, tuple) and all(isinstance(x, dict) for x in valid_outputs):
                    valid_outputs_format = valid_outputs[1]
                    valid_outputs = valid_outputs[0]
                    valid_outputs_format = {k: (valid_outputs_format[k] if k in valid_outputs_format.keys() else '') for k in valid_outputs.keys()}
                else:
                    raise TypeError('valid output must be either dict(name : val) for 2-tuple (dict(name : val), dict(name : format))')
            else:
                valid_outputs, valid_outputs_format = {}, {}

            # print epoch stat
            frmt = f"0{len(str(n_epochs))}"
            self.print_fn(f"Epoch {epoch+1:{frmt}}/{n_epochs:{frmt}} | "
                          f"Time {timedelta(seconds=time.time()-epoch_start_time)} | "
                          + "".join([f"{name} {loss_i / n_batches:.5f} | " for name, loss_i in train_losses.items()])
                          + "".join([f"{name} {val:{valid_outputs_format[name]}} | " for name, val in valid_outputs.items()]))

            epoch_loss_list.append([epoch+1,
                                    {name : loss/n_batches for name, loss in train_losses.items()},
                                    valid_outputs])

            # scheduler steps
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Update best model
            if save_best_key:
                assert save_best_key in valid_outputs.keys(), f"`save_best_key` must be present in the validation output dict to save the best model."
                # initialize if first epoch
                if epoch == 0:
                    best_metric = valid_outputs[save_best_key]
                    best_epoch = epoch+1
                    self.best_net = copy.deepcopy(self.net)
                # update best net
                if (minimize_metric and valid_outputs[save_best_key] < best_metric) \
                or (not minimize_metric and valid_outputs[save_best_key] > best_metric) \
                or epoch < min_epoch_best:
                    best_metric = valid_outputs[save_best_key]
                    best_epoch = epoch+1
                    self.best_net = copy.deepcopy(self.net)

            # Save checkpoint
            if (epoch+1) % checkpoint_freq == 0 and checkpoint_path is not None:
                checkpoint = {
                    'n_epoch_finished': epoch+1,
                    'net_state': self.net.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'loss_evolution': epoch_loss_list
                }
                if save_best_key:
                    checkpoint['best_metric'] = best_metric
                    checkpoint['best_epoch'] = best_epoch
                    checkpoint['best_net_state'] = self.best_net.state_dict()
                if self.lr_scheduler:
                    checkpoint['lr_state'] = self.lr_scheduler.state_dict()

                for k in self.extra_checkpoint_keys:
                    checkpoint[k] = getattr(self, k)

                torch.save(checkpoint, checkpoint_path)
                self.print_fn('\tCheckpoint saved.')

        self.outputs['train_evolution'] = epoch_loss_list
        if save_best_key:
            self.outputs['best_model'] = {save_best_key : best_metric, 'epoch' : best_epoch}

    def validate(self, loader: DataLoader, epoch: int = None) -> Dict:
        """
        --> Define how to validate the model. It should return a dictionnary with relevant validation metrics to be
        printed in the training evolution or to select the best model. If no metrics should be return, you should retrun
        an empty dict `{}`. Addtionnally, a second dictionnary can be returned that defined the formating of each/some
        entries of the other dictionnary.

        output : Dictionnary {"Name" : Value} e.g. {"Accuracy" : 0.875, "Loss" : 0.03421, ...}
                 (Dictionnary {"Name" : str_format} e.g. {"Accuracy" : ".2%", ...})
        """
        raise NotImplementedError("self.validate(loader) must be implemented when a valid Dataloader is passed to self.train().")

    @abstractmethod
    def forward_loss(self, data: Tuple[Tensor]) -> Tensor:
        """
        --> Define Forward + Loss Computation from data provided by loader
        Can return a tuple (loss, {'Loss_name1': sub_loss1, 'Loss_name2': sub_loss2, ...}) or just the loss to backward on.
        --> if dictionnary is returned as a second element, each sub-loss will be displayed in epoch summary.

        e.g.
        inputs, labels = data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.net(inputs)
        return loss_fn(outputs, labels)
        """
        pass

    def load(self, import_path: str, map_location: str = 'cuda:0'):
        """
        Load the model state dictionnary at the import path on the device specified by map_location.
        """
        loaded_state_dict = torch.load(import_path, map_location=map_location)
        self.net.load_state_dict(loaded_state_dict)

    def save(self, export_path: str):
        """
        Save model state dictionnary at the export_path.
        """
        torch.save(self.net.state_dict(), export_path)

    def save_best(self, export_path : str):
        """
        Save the best model state dictionnary at the export_path.
        """
        torch.save(self.best_net.state_dict(), export_path)

    def transfer_weight(self, import_path: str, map_location: str = 'cuda:0', verbose: bool = True):
        """
        Transfer all matching keys of the model state dictionnary at the import path to self.net.
        """
        # load pretrain weights
        init_state_dict = torch.load(import_path, map_location=map_location)
        # get self.net state dict
        net_state_dict = self.net.state_dict()
        # get common keys
        to_transfer_keys = {k:w for k, w in init_state_dict.items() if k in net_state_dict}
        if verbose:
            self.print_fn(f'{len(to_transfer_keys)} matching weight keys found on {len(init_state_dict)} to be tranferred to the net ({len(net_state_dict)} weight keys).')
        # update U-Net weights
        net_state_dict.update(to_transfer_keys)
        self.net.load_state_dict(net_state_dict)

class BaseModelDualNet(BaseModel):
    def __init__(self, net1: nn.Module, net2: nn.Module, opt1: Optimizer = None, opt2: Optimizer = None,
                 sched1: _LRScheduler = None, sched2: _LRScheduler = None, grad_scaler: GradScaler = None,
                 logger: Logger = None, print_progress: bool = True, device: str = 'cuda:0', **kwargs):
        """

        """
        super().__init__(logger=logger, print_progress=print_progress, device=device, **kwargs)

        self.net1, self.net2 = net1, net2
        self.net1, self.net2 = self.net1.to(device), self.net2.to(device)
        self.best_net1, self.best_net2 = net1, net2
        self.best_metric = None
        self.opt1, self.opt2 = opt1, opt2
        self.lr_scheduler1, self.lr_scheduler2 = sched1, sched2
        self.grad_scaler = grad_scaler

        self.logger = logger
        self.device = device
        self.print_progress = print_progress

        self.outputs = {} # placeholder for any output to be saved in YAML

    def train(self, n_epochs: int, train_loader: DataLoader, valid_loader: DataLoader = None,
              extra_valid_args: List = [], extra_valid_kwargs: Dict = dict(),
              checkpoint_path: str = None, checkpoint_freq: int = 10,
              save_best_key : str = None, minimize_metric : bool = True, min_epoch_best : int = 0):
        """
        Train the self.net using the optimizer and scheduler using the data provided by the train_loader. At each epoch,
        the model can be validated using the valid_loader (if a valid loader is provided, the method self.validate must
        be implemented in the children). The model and training state is loaded/saved in a .pt file if checkpoint_path
        is provided. The model is then saved every checkpoint_freq epoch.

        The best model can be saved over the training processed based on one of the validation metric provided by the
        self.validate output dictionnary. The metric to use is specified by the string `save_best_key` and the argument
        `minimize_metric` define whether the metric must be minimized or maximized. A mininumm number of epoch to be
        performed before selcting the best model can be specified with 'min_epoch_best'.
        """
        #assert self.opt1 is not None, "At least an optimizer must be provided to train the model."

        # Load checkpoint if any
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                n_epoch_finished = checkpoint['n_epoch_finished']
                self.net1.load_state_dict(checkpoint['net1_state'])
                self.net1 = self.net1.to(self.device)
                self.net2.load_state_dict(checkpoint['net2_state'])
                self.net2 = self.net2.to(self.device)
                if self.opt1:
                    self.opt1.load_state_dict(checkpoint['optimizer1_state'])
                if self.opt2:
                    self.opt2.load_state_dict(checkpoint['optimizer2_state'])
                if save_best_key:
                    best_metric = checkpoint['best_metric']
                    best_epoch = checkpoint['best_epoch']
                    self.best_net1.load_state_dict(checkpoint['best_net1_state'])
                    self.best_net1 = self.best_net1.to(self.device)
                    self.best_net2.load_state_dict(checkpoint['best_net2_state'])
                    self.best_net2 = self.best_net2.to(self.device)

                if self.lr_scheduler1:
                    self.lr_scheduler1.load_state_dict(checkpoint['lr1_state'])
                if self.lr_scheduler2:
                    self.lr_scheduler2.load_state_dict(checkpoint['lr2_state'])

                if self.grad_scaler:
                    self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

                epoch_loss_list = checkpoint['loss_evolution']

                for k in self.extra_checkpoint_keys:
                    setattr(self, k, checkpoint[k])

                self.print_fn(f'Resuming from Checkpoint with {n_epoch_finished} epoch finished.')
            except FileNotFoundError:
                self.print_fn('No Checkpoint found. Training from beginning.')
                n_epoch_finished = 0
                epoch_loss_list = [] # Placeholder for epoch evolution
        else:
            self.print_fn('No Checkpoint used. Training from beginning.')
            n_epoch_finished = 0
            epoch_loss_list = []

        self.net1 = self.net1.to(self.device)
        self.net2 = self.net2.to(self.device)
        self.nbatch_per_ep = len(train_loader)
        # Train Loop
        for epoch in range(n_epoch_finished, n_epochs):
            self.net1.train()
            self.net2.train()
            epoch_start_time = time.time()
            epoch_loss = 0.0
            n_batches = len(train_loader)

            for b, data in enumerate(train_loader):
                # Processing of one batch
                loss_outputs = self.process_batch(epoch, b, data)

                # Recover the losses and potential display formating
                if loss_outputs is not None:
                    # unpack format dictionnary if provided
                    if isinstance(loss_outputs, dict):
                        loss_outputs = {key : '' for key in valid_outputs.keys()}
                    elif isinstance(loss_outputs, tuple) and all(isinstance(x, dict) for x in loss_outputs):
                        loss_outputs_format = loss_outputs[1]
                        loss_outputs = loss_outputs[0]
                        loss_outputs_format = {k: (loss_outputs_format[k] if k in loss_outputs_format.keys() else '') for k in loss_outputs.keys()}
                    else:
                        raise TypeError('loss output must be either dict(name : val) for 2-tuple (dict(name : val), dict(name : format))')
                else:
                    loss_outputs, loss_outputs_format = {}, {}

                # Store the batch loss in the total epoch loss
                if b == 0:
                    sum_loss_outputs = {name : 0.0 for name in loss_outputs.keys()}
                sum_loss_outputs = {name : (value + loss_outputs[name].item() if isinstance(loss_outputs[name], torch.Tensor) \
                                           else value + loss_outputs[name]) \
                                    for name, value in sum_loss_outputs.items()}

                if self.print_progress:
                    self.print_progessbar(b, n_batches, name='Train Batch', size=100, erase=True)

            # validate epoch
            if valid_loader:
                self.net1.eval()
                self.net2.eval()
                valid_outputs = self.validate(valid_loader, epoch, *extra_valid_args, **extra_valid_kwargs)
                # unpack format dictionnary if provided
                if isinstance(valid_outputs, dict):
                    valid_outputs_format = {key : '' for key in valid_outputs.keys()}
                elif isinstance(valid_outputs, tuple) and all(isinstance(x, dict) for x in valid_outputs):
                    valid_outputs_format = valid_outputs[1]
                    valid_outputs = valid_outputs[0]
                    valid_outputs_format = {k: (valid_outputs_format[k] if k in valid_outputs_format.keys() else '') for k in valid_outputs.keys()}
                else:
                    raise TypeError('valid output must be either dict(name : val) for 2-tuple (dict(name : val), dict(name : format))')
            else:
                valid_outputs, valid_outputs_format = {}, {}

            # print epoch stat
            frmt = f"0{len(str(n_epochs))}"
            self.print_fn(f"Epoch {epoch+1:{frmt}}/{n_epochs:{frmt}} | "
                          f"Time {timedelta(seconds=time.time()-epoch_start_time)} | "
                          + "".join([f"{name} {loss_i / n_batches:{loss_outputs_format[name]}} | " for name, loss_i in sum_loss_outputs.items()])
                          + "".join([f"{name} {val:{valid_outputs_format[name]}} | " for name, val in valid_outputs.items()]))

            epoch_loss_list.append([epoch+1,
                                    {name : loss/n_batches for name, loss in sum_loss_outputs.items()},
                                    valid_outputs])

            # scheduler steps
            if self.lr_scheduler1:
                self.lr_scheduler1.step()
            if self.lr_scheduler2:
                self.lr_scheduler2.step()

            # Update best model
            if save_best_key:
                assert save_best_key in valid_outputs.keys(), f"`save_best_key` must be present in the validation output dict to save the best model."
                # initialize if first epoch
                if epoch == 0:
                    best_metric = valid_outputs[save_best_key]
                    best_epoch = epoch+1
                    self.best_net1 = copy.deepcopy(self.net1)
                    self.best_net2 = copy.deepcopy(self.net2)
                # update best net
                if (minimize_metric and valid_outputs[save_best_key] < best_metric) \
                or (not minimize_metric and valid_outputs[save_best_key] > best_metric) \
                or epoch < min_epoch_best:
                    best_metric = valid_outputs[save_best_key]
                    best_epoch = epoch+1
                    self.best_net1 = copy.deepcopy(self.net1)
                    self.best_net2 = copy.deepcopy(self.net2)

            # Save checkpoint
            if (epoch+1) % checkpoint_freq == 0 and checkpoint_path is not None:
                checkpoint = {
                    'n_epoch_finished': epoch+1,
                    'net1_state': self.net1.state_dict(),
                    'net2_state': self.net2.state_dict(),
                    'loss_evolution': epoch_loss_list
                }
                if save_best_key:
                    checkpoint['best_metric'] = best_metric
                    checkpoint['best_epoch'] = best_epoch
                    checkpoint['best_net1_state'] = self.best_net1.state_dict()
                    checkpoint['best_net2_state'] = self.best_net2.state_dict()
                if self.opt1:
                    checkpoint['optimizer1_state'] = self.opt1.state_dict()
                if self.opt2:
                    checkpoint['optimizer2_state'] = self.opt2.state_dict()
                if self.lr_scheduler1:
                    checkpoint['lr1_state'] = self.lr_scheduler1.state_dict()
                if self.lr_scheduler2:
                    checkpoint['lr2_state'] = self.lr_scheduler2.state_dict()
                if self.grad_scaler:
                    checkpoint['grad_scaler_state'] = self.grad_scaler.state_dict()

                for k in self.extra_checkpoint_keys:
                    checkpoint[k] = getattr(self, k)

                torch.save(checkpoint, checkpoint_path)
                self.print_fn('\tCheckpoint saved.')

        self.outputs['train_evolution'] = epoch_loss_list
        if save_best_key:
            self.outputs['best_model'] = {save_best_key : best_metric, 'epoch' : best_epoch}

    def validate(self, loader: DataLoader, epoch: int = None, *args, **kwargs) -> Dict:
        """
        --> Define how to validate the model. It should return a dictionnary with relevant validation metrics to be
        printed in the training evolution or to select the best model. If no metrics should be return, you should retrun
        an empty dict `{}`. Addtionnally, a second dictionnary can be returned that defined the formating of each/some
        entries of the other dictionnary.

        output : Dictionnary {"Name" : Value} e.g. {"Accuracy" : 0.875, "Loss" : 0.03421, ...}
                 (Dictionnary {"Name" : str_format} e.g. {"Accuracy" : ".2%", ...})
        """
        raise NotImplementedError("self.validate(loader) must be implemented when a valid Dataloader is passed to self.train().")

    @abstractmethod
    def process_batch(self, epoch: int, b: int, data: Tuple[Tensor]) -> Tensor:
        """
        Define the processing of a single batch (data extraction, forward, backward and optimizier steps).
        also receive the current epoch number and the current batch number.

        return either:
            - None
            - one dictionnary of different losses {'name' : loss} with loss either as float or Tensor
            - two dictionnary: one for the different losses {'name' : loss_val, ...} with loss either as float or Tensor,
                               one for the display formating given as {'name': '.4f', ...}
        """
        pass

    def save(self, export_path: str, which: str = 'both', best: bool = False):
        """

        """
        assert which in ['both', '1', '2'], f"'which' must be either 'both', '1', or '2'. Given: {which}."
        if which == 'both':
            assert isinstance(export_path, list), f"When 'which' is 'both', 'export_path' must be a list of two strings."
            assert len(export_path) == 2, f"'export_path' must be a list of length 2. Given length: {len(export_path)}"

        if which == 'both':
            torch.save(self.best_net1.state_dict() if best else self.net1.state_dict(), export_path[0])
            torch.save(self.best_net2.state_dict() if best else self.net2.state_dict(), export_path[1])
        elif which == '1':
            torch.save(self.best_net1.state_dict() if best else self.net1.state_dict(), export_path)
        elif which == '2':
            torch.save(self.best_net2.state_dict() if best else self.net2.state_dict(), export_path)

    def load(self, import_path: str, map_location: str = 'cuda:0', which: str = 'both', best: bool = False):
        """

        """
        assert which in ['both', '1', '2'], f"'which' must be either 'both', '1', or '2'. Given: {which}."
        if which == 'both':
            assert isinstance(import_path, list), f"When 'which' is 'both', 'import_path' must be a list of two strings."
            assert len(import_path) == 2, f"'import_path' must be a list of length 2. Given length: {len(import_path)}"

        if which == 'both':
            loaded_state_dict1 = torch.load(import_path[0], map_location=map_location)
            loaded_state_dict2 = torch.load(import_path[1], map_location=map_location)
            if best:
                self.best_net1.load_state_dict(loaded_state_dict1)
                self.best_net2.load_state_dict(loaded_state_dict2)
            else:
                self.net1.load_state_dict(loaded_state_dict1)
                self.net2.load_state_dict(loaded_state_dict2)

        elif which == '1':
            loaded_state_dict = torch.load(import_path, map_location=map_location)
            if best:
                self.best_net1.load_state_dict(loaded_state_dict)
            else:
                self.net1.load_state_dict(loaded_state_dict)

        elif which == '2':
            loaded_state_dict = torch.load(import_path, map_location=map_location)
            if best:
                self.best_net2.load_state_dict(loaded_state_dict)
            else:
                self.net2.load_state_dict(loaded_state_dict)

    # def transfer_weight(self, import_path: str, map_location: str = 'cuda:0', verbose: bool = True):
    #     """
    #     Transfer all matching keys of the model state dictionnary at the import path to self.net.
    #     """
    #     # load pretrain weights
    #     init_state_dict = torch.load(import_path, map_location=map_location)
    #     # get self.net state dict
    #     net_state_dict = self.net.state_dict()
    #     # get common keys
    #     to_transfer_keys = {k:w for k, w in init_state_dict.items() if k in net_state_dict}
    #     if verbose:
    #         self.print_fn(f'{len(to_transfer_keys)} matching weight keys found on {len(init_state_dict)} to be tranferred to the net ({len(net_state_dict)} weight keys).')
    #     # update U-Net weights
    #     net_state_dict.update(to_transfer_keys)
    #     self.net.load_state_dict(net_state_dict)
