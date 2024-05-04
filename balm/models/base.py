#!/usr/bin/python
# filename: base_model.py

#
# Copyright (c) 2024 Bryan Briney
# License: GNU General Public License, version 3.0 (http://opensource.org/licenses/gpl-3-0/)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


import os
from typing import Optional, Union

import torch
import torch.nn as nn

from ..config import BaseConfig


class BalmBase(nn.Module):
    """
    Base class for Balm models.
    """

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(
        cls, model_path: str, config: Optional[Union[str, BaseConfig, dict]] = None
    ):
        """
        Load a pretrained model.

        Parameters
        ----------
        model_path : str
            The path to a diretory containing the pretrained model.
            Model file should be named ``"model.pt"``.

        config : Optional[Union[str, BaseConfig, dict]], optional
            Alternate configuration object or path to an alternate configuration file.
            If not provided, the configuration (``config.json``) will be loaded from the
            model directory.

        Returns
        -------
        cls
            The loaded model.
        """
        # config
        if config is None:
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Configuration file not found in the model directory: {model_path}"
                )
            config = cls.config_cls.from_json(config_path)
        elif isinstance(config, str):
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            config = cls.config_cls.from_json(config_path)
        elif isinstance(config, BaseConfig):
            pass
        elif isinstance(config, dict):
            config = cls.config_cls.from_dict(config)
        else:
            raise ValueError(f"Invalid configuration type: {type(config)}")

        # model
        model_path = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file (model.pt) not found in the supplied model directory: {model_path}"
            )
        model = cls(config=config)
        model.load_state_dict(torch.load(model_path))
        return model
