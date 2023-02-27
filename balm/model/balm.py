#!/usr/bin/python
# filename: balm.py

#
# Copyright (c) 2023 Bryan Briney
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


import sys
from typing import Union

from transformers import RobertaConfig, RobertaForMaskedLM


__all__ = ["get_model", "BALM_PARAMS"]


def get_model(model_config: Union[str, RobertaConfig, None] = None):
    """
    Retrieves a BALM model, given a model configuration.

     Parameters
     ----------
     model_config : Union[str, RobertaConfig, None], default=None
         Can be either a ``RobertaConfig`` object, or the name of a built-in
         model. Built-in model options are:
             * ``"balm"``: default BALM model size, based on RoBERTa-large
             * ``"balm-small"``: reduced size BALM model, based on RoBERTa
             * ``"balm-large"``: based on LLaMA 7B
             * ``"balm-xlarge"``: based on LLaMA 13B
             * ``"balm-huge"``: based on LLaMA 33B
             * ``"balm-gigantic"``: based on LLaMA 65B
         Default is ``"balm"``.

     Returns
     -------
     RobertaConfig
         Model configuration
    """
    if isinstance(model_config, RobertaConfig):
        config = model_config
    else:
        if model_config is None:
            model_config = "balm"
        if model_config not in BALM_PARAMS:
            err = "\nERROR: Invalid model name. Options are:\n  -"
            err += "\n  - ".join(BALM_PARAMS.keys())
            err += "\n\n"
            print(err)
            sys.exit(1)
        params = BALM_PARAMS[model_config.lower()]
        config = RobertaConfig(**params)
    return RobertaForMaskedLM(config)


BALM_PARAMS = {
    # based on RoBERTa-large
    "balm": {
        "vocab_size": 25,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "max_position_embeddings": 320,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "type_vocab_size": 2,
    },
    # based on RoBERTa
    "balm-small": {
        "vocab_size": 25,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "max_position_embeddings": 320,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "type_vocab_size": 2,
    },
    # based on LLaMA 7B
    "balm-large": {
        "vocab_size": 25,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "max_position_embeddings": 320,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "type_vocab_size": 2,
    },
    # based on LLaMA 13B
    "balm-xlarge": {
        "vocab_size": 25,
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "max_position_embeddings": 320,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "type_vocab_size": 2,
    },
    # based on LLaMA 33B
    "balm-huge": {
        "vocab_size": 25,
        "hidden_size": 1664,
        "intermediate_size": 6656,
        "max_position_embeddings": 320,
        "num_hidden_layers": 52,
        "num_attention_heads": 60,
        "type_vocab_size": 2,
    },
    # based on LLaMA 65B
    "balm-gigantic": {
        "vocab_size": 25,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "max_position_embeddings": 320,
        "num_hidden_layers": 64,
        "num_attention_heads": 80,
        "type_vocab_size": 2,
    },
}
