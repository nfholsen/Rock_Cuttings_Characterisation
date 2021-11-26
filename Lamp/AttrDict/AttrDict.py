from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Sequence, Union
import os
import logging
from logging import Logger
import yaml
import json

class AttrDict(dict):
    """
    Dictionary subclass whose entries can be accessed like attributes
    (as well as normally).
    """
    def __init__(self, *args, **kwargs):
        """
        Build a AttrDict from dict like this : AttrDict.from_nested_dicts(dict)
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_json_path(path : str):# -> AttrDict:
        """ Construct nested AttrDicts from a json. """
        assert os.path.isfile(path), f'Path {path} does not exist.'
        with open(path, 'r') as fn:
            data = json.load(fn)
        return AttrDict.from_nested_dicts(data)

    @staticmethod
    def from_yaml_path(path : str, loader: yaml.Loader = yaml.SafeLoader):# -> AttrDict:
        """ Construct nested AttrDicts from a YAML path with the specified yaml.loader. """
        assert os.path.isfile(path), f'Path {path} does not exist.'
        with open(path, 'r') as fn:
            data = yaml.load(fn, Loader=loader)
        return AttrDict.from_nested_dicts(data)

    def to_yaml(self, path : str):
        """ Save the nested AttrDicts (self) in a YAML file specified by path """
        assert os.path.isdir(os.path.dirname(path)), f'Path {os.path.dirname(path)} does not exist.'
        with open(path, 'w') as fn:
            yaml.dump(self.to_nested_dicts(self), fn, sort_keys=False)

    @staticmethod
    def as_json_proof(d : dict):
        """ Convert a dictionnary to a JSON serializable one by converting non-native type to string. """
        def is_jsonable(x):
            try:
                json.dumps(x)
                return True
            except (TypeError, OverflowError):
                return False

        if isinstance(d, dict):
            return {k: as_json_proof(v) for k, v in d.items()}
        else:
            if not is_jsonable(d):
                return str(d)
            else:
                return d

    def to_json(self, path : str):
        """ Save the nested AttrDicts (self) in a JSON file specified by path """
        assert os.path.isdir(os.path.dirname(path)), f'Path {os.path.dirname(path)} does not exist.'
        with open(path, 'w') as fn:
            json.dump(self.as_json_proof(self), fn)

    @staticmethod
    def from_nested_dicts(data : dict):# -> AttrDict:
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_nested_dicts(data[key]) for key in data})

    @staticmethod
    def to_nested_dicts(data):# -> AttrDict:
        """ Construct nested dict from an AttrDict. """
        if not isinstance(data, AttrDict):
            return data
        else:
            return dict({key: AttrDict.to_nested_dicts(data[key]) for key in data})