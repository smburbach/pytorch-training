#!/usr/bin/python
# filename: balm_switch.py

import sys
from typing import Union

from transformers import SwitchTransformersConfig, SwitchTransformersModel


__all__ = ["get_model", "BALM_PARAMS"]


def get_model(model_config: Union[str, SwitchTransformersConfig, None] = None):
    """
    Retrieves a BALM model, given a model configuration.

     Parameters
     ----------
     model_config : Union[str, SwitchTransformersConfig, None], default=None
         Can be either a ``SwitchTransformersConfig`` object, or the name of a built-in
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
     SwitchTransformersConfig
         Model configuration
    """
    if isinstance(model_config, SwitchTransformersConfig):
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
        config = SwitchTransformersConfig(**params)
    return SwitchTransformersModel(config)


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
