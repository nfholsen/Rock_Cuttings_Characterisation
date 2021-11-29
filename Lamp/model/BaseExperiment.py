from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Sequence, Union
import os
import logging
from logging import Logger
import yaml
import json

import torch
import numpy as np
import random

class BaseExperiment(ABC):
    def __init__(self, cfg_path):
        """
        Basic methods for general experiment wrapped in an object. The base experiement takes as input a path to a
        config file of type AttrDict. It has an attribute named 'required_entries' that allow define some required keys
        in the config file as well as their allowed type. 'required_entries' is a nested dictionnary containing
        'key name':[type1, type2, ...] or 'key name':{'nested key name': [type1, type2, ...]} and so on. The config
        file is loaded in an attribute self.cfg and keys cab be accessed using the '.' such as 'cfg.attr1.subattr2'.
        Presence of the required entries can be controled using the self.check_cfg_entries
        """
        # check cfg_integrity {key:list(type)}
        self.required_entries = {'exp_name':[str], "save_dir": [str], "device": [str, type(None)]}
        # load cfg
        self.cfg = self.load_config(cfg_path)
        # check config entries
        self.check_cfg_entries(self.cfg, self.required_entries)
        self.print_fn = print

    def check_cfg_entries(self, cfg : AttrDict, entry_dict: Dict, parent_entry : Sequence[str] = None):
        """ Check that the keys in entry_dict are found in the passed cfg with the specified type."""
        for key, type_list in entry_dict.items():
            if parent_entry is None:
                parent_entry = []
            # check presence
            try:
                _ = cfg[key]
            except KeyError:
                raise(KeyError(f" cfg.{'.'.join(parent_entry+[''])}{key} could not be found"))
            # check type
            if isinstance(type_list, dict):
                self.check_cfg_entries(cfg[key], type_list, parent_entry=parent_entry+[key])
            else:
                assert isinstance(cfg[key], tuple(type_list)), f"Type Mismatch. Config file entry cfg.{'.'.join(parent_entry+[''])}{key} must be one of {type_list}. Got {type(cfg[key])}."

    def print_cfg(self, cfg : AttrDict, prefix : str = '|-'):
        for k, v in cfg.items():
            if isinstance(v, dict):
                self.print_fn(f"{prefix} {k}")
                self.print_cfg(v, prefix=prefix+'--')
            else:
                self.print_fn(f"{prefix} {k} -> {v}")

    def load_config(self, cfg_path):
        """  """
        if os.path.splitext(cfg_path)[-1] == '.json':
            return AttrDict.from_json_path(cfg_path)
        elif os.path.splitext(cfg_path)[-1] in ['.yaml', '.yml']:
            return AttrDict.from_yaml_path(cfg_path)
        else:
            raise ValueError(f"Unsupported config file format. Only '.json', '.yaml' and '.yml' files are supported.")

    def creat_exp_dir(self):
        """  """
        self.exp_dir = os.path.join(self.cfg.save_dir, self.cfg.exp_name)
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir, exist_ok=False)

    @staticmethod
    def initialize_logger(logger_fn):
        """ Initialize a logger with given file name. It will start a new logger. """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        try:
            logger.handlers[1].stream.close()
            logger.removeHandler(logger.handlers[1])
        except IndexError:
            pass
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(logger_fn)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(file_handler)

        return logger

    @staticmethod
    def set_seed(seed):
        """ Set the random seed """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #torch.backends.cudnn.deterministic = True

    def set_device(self):
        """  """
        # either set device if string or select automatically GPU is available (check that entry are either 'cpu', 'cuda', 'cuda:{i}' or 'auto')
        if isinstance(self.cfg.device, str):
            self.cfg.device = torch.device(self.cfg.device)
        else:
            self.cfg.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def save_config(self, save_path):
        """  """
        format = os.path.splitext(save_path)[-1]
        if format == '.json':
            self.cfg.to_json(save_path)
        elif format in ['.yaml', '.yml']:
            self.cfg.to_yaml(save_path)
        else:
            raise ValueError(f"Only '.json', '.yaml', '.yml' are supported as format to save the config. Got '{format}'.")

    @abstractmethod
    def run(self, *args, **kwargs):
        """ Define the experiment steps. """
        pass

class BaseReplicateExperiment(BaseExperiment):
    def __init__(self, cfg_path, log=True):
        """  """
        super().__init__(cfg_path)
        self.required_entries.update({"seed": [int, float, list]})
        self.log = log
        # control config
        self.check_cfg_entries(self.cfg, self.required_entries)

    def run(self, *args, **kwargs):
        """  """
        # make exp_dir
        self.creat_exp_dir()
        # initialize print_fn
        if self.log:
            self.logger = self.initialize_logger(os.path.join(self.exp_dir, 'log.txt'))
            self.print_fn = self.logger.info
        else:
            self.print_fn = print

        self.print_fn("="*30)
        self.print_fn(f"\tStarting Experiment {self.cfg.exp_name}.")
        self.print_fn("="*30)
        # print config
        self.print_cfg(self.cfg)
        # set device
        self.set_device()
        self.print_fn(f"Device set to '{self.cfg.device}'.")
        # if seed is list
        seed_list = self.cfg.seed if isinstance(self.cfg.seed, list) else [self.cfg.seed]
        # replicate experiment for each seed
        for i, seed in enumerate(seed_list):
            # make seed folder
            if len(seed_list) == 1:
                self.replicate_dir = self.exp_dir
            else:
                self.replicate_dir = os.path.join(self.exp_dir, f"replicate{i+1}_seed{seed}")
            if not os.path.isdir(self.replicate_dir):
                os.mkdir(self.replicate_dir)
            # Check if replicate has already been processed (when recovering experiement)
            if not os.path.isfile(os.path.join(self.replicate_dir, '.done.txt')):
                # fix the seed
                self.set_seed(seed)
                frmt = f"0{len(str(len(seed_list)))}"
                self.print_fn("="*30)
                self.print_fn(f"\tExperiment Replicate {i+1:{frmt}}/{len(seed_list):{frmt}} with seed {seed_list[i]}")
                self.print_fn("="*30)
                # run replicate
                self.run_replicate(*args, **kwargs)
                # save a hidden file to indicate that the replicate has been runned and finished
                with open(os.path.join(self.replicate_dir, '.done.txt'), 'w') as f:
                    f.write('')
            else:
                frmt = f"0{len(str(len(seed_list)))}"
                self.print_fn(f"\tSkipping Experiment Replicate {i+1:{frmt}}/{len(seed_list):{frmt}} as it already exists.")

        # remove the hidden .done.txt once all replicates are runned
        # for i, seed in enumerate(seed_list):
        #     if len(seed_list) == 1:
        #         self.replicate_dir = self.exp_dir
        #     else:
        #         self.replicate_dir = os.path.join(self.exp_dir, f"replicate{i+1}_seed{seed}")
        #     os.remove(os.path.join(self.replicate_dir, '.done.txt'))

        # save config
        cfg_save_path = os.path.join(self.exp_dir, 'config.yaml')
        self.save_config(cfg_save_path)
        self.print_fn(f"Configuration file saved at {cfg_save_path}")

    @abstractmethod
    def run_replicate(self, *args, **kwargs):
        """ Define how to run a single replicate of experiemnt. """
        pass

class BaseCrossValExperiment(BaseExperiment):
    def __init__(self, cfg_path, log=True):
        """  """
        super().__init__(cfg_path)
        pass

    def run(self, *args, **kwargs):
        """  """
        pass

    @abstractmethod
    def run_fold(self, fold, *args, **kwargs):
        """  """
        pass
