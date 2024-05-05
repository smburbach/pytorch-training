#!/usr/bin/python
# filename: balm_moe_config.py

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


from typing import Optional

from .base_config import BaseConfig


class BalmExpertChoiceMoEConfig(BaseConfig):
    def __init__(
        self,
        embed_dim: int = 320,
        ffn_dim: int = 1280,
        num_layers: int = 6,
        num_heads: int = 20,
        num_experts: int = 8,
        max_length: int = 320,
        vocab_size: int = 33,
        expert_capacity: Optional[int] = None,
        expert_capacity_multiplier: float = 1.5,
        num_shared_experts: int = 0,
        expert_activation: str = "gelu",
        router_z_loss_coef: float = 0.001,
        alternate_sparsity: bool = False,
        router_top_k: int = 1,
        router_bias: bool = False,
        router_jitter: float = 0.0,
        router_dtype: str = "float32",
        router_ignore_padding_tokens: bool = True,
        attention_dropout: float = 0.0,
        expert_ffn_dropout: float = 0.0,
        token_embedding_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        padding_idx: int = 0,
    ):
        """
        Configuration for the BalmExpertChoiceMoE model.

        Parameters
        ----------
        embed_dim : int, default=320
            The dimension of the token embeddings.

        ffn_dim : int, default=1280
            The dimension of the feed-forward network.

        num_layers : int, default=6
            The number of layers in the transformer.

        num_heads : int, default=20
            The number of heads in the transformer.

        num_experts : int, default=8
            The number of experts in the transformer.

        max_length : int, default=320
            The maximum length of the input sequence.

        vocab_size : int, default=33
            The vocabulary size.

        expert_capacity : int, optional
            The capacity of each expert. If not provided, it will be calculated as `max_length / num_experts * expert_capacity_multiplier`.

        expert_capacity_multiplier : float, default=1.25
            The multiplier for the expert capacity.

        expert_activation : str, default="gelu"
            The activation function to use for the experts. Options are "relu" and "gelu".

        router_z_loss_coef : float, default=0.001
            The coefficient for the router z loss.

        router_aux_loss_coef : float, default=0.001
            The coefficient for the router auxiliary loss.

        alternate_sparsity : bool, default=False
            Whether to use alternate sparsity for the router.

        router_top_k : int, default=1
            The top k to use for the router.

        router_bias : bool, default=False
            Whether to use a bias for the router.

        router_jitter : float, default=0.0
            The jitter to use for the router.

        router_dtype : str, default="float32"
            The dtype to use for the router. Options are "float32" and "float16".

        router_ignore_padding_tokens : bool, default=True
            Whether to ignore padding tokens for the router.

        attention_dropout : float, default=0.0
            The dropout to use for the attention.

        expert_ffn_dropout : float, default=0.0
            The dropout to use for the expert feed-forward network.

        token_embedding_dropout : float, default=0.0
            The dropout to use for the token embeddings.

        layer_norm_eps : float, default=1e-5
            The epsilon to use for the layer normalization.

        padding_idx : int, default=0
            The index to use for the padding tokens.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.expert_capacity = (
            expert_capacity
            if expert_capacity is not None
            else int(max_length / num_experts * expert_capacity_multiplier)
        )
        self.expert_capacity_multiplier = expert_capacity_multiplier
        self.num_shared_experts = num_shared_experts
        self.expert_activation = expert_activation
        self.router_z_loss_coef = router_z_loss_coef
        self.alternate_sparsity = alternate_sparsity
        self.router_top_k = router_top_k
        self.router_bias = router_bias
        self.router_jitter = router_jitter
        self.router_dtype = router_dtype
        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.attention_dropout = attention_dropout
        self.expert_ffn_dropout = expert_ffn_dropout
        self.token_embedding_dropout = token_embedding_dropout
        self.layer_norm_eps = layer_norm_eps
        self.padding_idx = padding_idx
