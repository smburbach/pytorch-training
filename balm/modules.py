#!/usr/bin/python
# filename: modules.py

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


from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .activation import SwiGLU
from .embedding import RotaryPositionalEmbeddings
from .router import TopKRouter


class MaskedLMOutput:
    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attentions: Optional[torch.Tensor] = None,
        last_hidden_state: Optional[torch.Tensor] = None,
        router_logits: Optional[torch.Tensor] = None,
        router_z_loss: Optional[torch.Tensor] = None,
        router_aux_loss: Optional[torch.Tensor] = None,
        expert_indices: Optional[torch.Tensor] = None,
    ):
        self.logits = logits
        self.loss = loss
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.last_hidden_state = last_hidden_state
        self.router_logits = router_logits
        self.router_z_loss = router_z_loss
        self.router_aux_loss = router_aux_loss
        self.expert_indices = expert_indices

    def __getitem__(self, idx):
        if isinstance(idx, str) and hasattr(self, idx):
            return getattr(self, idx)
        return self.logits[idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, str):
            setattr(self, idx, value)
        else:
            raise ValueError(f"Invalid key: {idx}. Keys must be strings.")

    def as_tuple(self):
        output_attrs = [
            self.last_hidden_state,
            self.attentions,
            self.hidden_states,
            self.router_logits,
        ]
        return tuple([o for o in output_attrs if o is not None])


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_batch_first: bool = True,
        layer_norm_eps: float = 1e-5,
        activation: str = "gelu",
    ):
        """
        Transformer block with relative position embeddings and GELU activation.

        Parameters
        ----------
        embed_dim : int
            The input embedding dimension.

        heads : int
            The number of attention heads.

        forward_expansion : int
            The expansion factor for the feedforward network.

        max_len : int
            The maximum sequence length.

        dropout : float
            The dropout probability.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=attention_batch_first,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        # pre-norm
        residual = x
        x = self.norm1(x)

        # attention
        x, _ = self.attention(
            x,
            x,
            x,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        if need_weights:
            x, weights = x
        x = residual + self.dropout(x)

        # pre-norm
        residual = x
        x = self.norm2(x)

        # feedforward
        x = self.feed_forward(x)
        x = residual + self.dropout(x)

        if need_weights:
            return x, weights
        return x


class RoformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        max_len: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_batch_first: bool = True,
        layer_norm_eps: float = 1e-5,
    ):
        """
        Transformer block with rotary embeddings and SwiGLU activation.

        Parameters
        ----------
        embed_dim : int
            The input embedding dimension.

        heads : int
            The number of attention heads.

        forward_expansion : int
            The expansion factor for the feedforward network.

        max_len : int
            The maximum sequence length.

        dropout : float
            The dropout probability.
        """
        super().__init__()
        self.rotary_embedding = RotaryPositionalEmbeddings(embed_dim, max_len)

        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=attention_batch_first,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            SwiGLU(),
            nn.Linear(ffn_dim // 2, embed_dim),  # adjusted for SwiGLU
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        # pre-norm
        query_norm = self.norm1(query)
        key_norm = self.norm1(key)
        value_norm = self.norm1(value)

        # rotary embeddings
        query_rot = self.rotary_embedding(query_norm)
        key_rot = self.rotary_embedding(key_norm)
        value_rot = self.rotary_embedding(value_norm)

        # attention
        x, _ = self.attention(
            query_rot,
            key_rot,
            value_rot,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        if need_weights:
            x, weights = x
        x = query + self.dropout(x)

        # pre-norm
        residual = x
        x = self.norm2(x)

        # feedforward
        x = self.feed_forward(x)
        x = residual + self.dropout(x)

        if need_weights:
            return x, weights
        return x


class BalmLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.gelu = nn.GELU()

    def forward(self, features):
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias
        return x


class Expert(nn.Module):
    """
    Expert module for a Sparse Transformer layer.

    Parameters:
    -----------
    embed_dim : int
        Embedding dimension.

    ffn_embed_dim : int
        Feed-forward network embedding dimension. Typically 4x the embedding dimension.

    dropout_rate : float
        Dropout rate. The default is ``0.0``.

    activation : str, optional
        Activation function to use. One of "relu" or "gelu". The default is "gelu".
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout_rate: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.wi = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.wo = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input hidden states.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, sequence_length, embed_dim).
        """
        x = self.wi(x)
        x = self.activation(x)
        x = self.dropout(x)
        # as usual with Switch Transformers, we need to be careful with dtypes
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and x.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            x = x.to(self.wo.weight.dtype)
        x = self.wo(x)
        return x


class SparseMLP(nn.Module):
    """
    Implementation of the Switch Transformers Sparse MLP module.

    Parameters:
    -----------
    config : BalmMoEConfig
        Model configuration class with all the parameters of the model.
        Initializing with a config file does not load the weights associated with the model, only the
        configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    router_class : nn.Module, optional
        Router class to use. The default is ``TopKRouter``.

    expert_class : nn.Module, optional
        Expert class to use. The default is ``Expert``.

    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_experts: int,
        expert_capacity: int,
        num_shared_experts: int = 0,
        top_k: int = 1,
        expert_activation: str = "gelu",
        expert_ffn_dropout: float = 0.0,
        router_dtype: str = "float32",
        router_bias: bool = False,
        router_jitter: float = 0.0,
        router_ignore_padding_tokens: bool = True,
        router_class: nn.Module = TopKRouter,
        expert_class: nn.Module = Expert,
    ):
        super().__init__()
        self.router = router_class(
            embed_dim=embed_dim,
            num_experts=num_experts,
            expert_capacity=expert_capacity,
            top_k=top_k,
            num_shared_experts=num_shared_experts,
            dtype=router_dtype,
            bias=router_bias,
            jitter=router_jitter,
            ignore_padding_tokens=router_ignore_padding_tokens,
        )
        self.experts = nn.ModuleDict()
        for idx in range(num_experts):
            self.experts[f"expert_{idx}"] = expert_class(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                dropout_rate=expert_ffn_dropout,
                activation=expert_activation,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Route tokens to experts and process them.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, sequence_length, embed_dim).
        """
        # get the router mask, probabilities, and logits
        expert_mask, router_probs, router_logits = self.router(x)
        expert_outputs = []

        for idx, expert in self.experts.items():
            int_idx = int(idx.split("_")[-1])
            token_indices = expert_mask[..., int_idx].bool()
            expert_output = expert(x[token_indices]).to(x.dtype)
            expanded_output = torch.zeros_like(x)
            expanded_output[token_indices] = expert_output
            expert_outputs.append(expanded_output)

        # Combine the outputs from the selected tokens for each expert
        x = torch.stack(expert_outputs, dim=-1) * expert_mask.unsqueeze(-2)
        x = x.sum(dim=-1)

        return x, (router_logits, expert_mask)


class SparseTransformerLayer(nn.Module):
    """
    BALM transformer layer with Mixture of Experts. Approximately follows the ESM-2
    implementation, but differs in a few ways:
        - includes (optional) dropout for self-attention and feedforward layers
        - normalize **after**, not before, the self-attention and feedforward layers
        - we don't use rotary embeddings, which aren't (yet?) compatible with
          torch's optimized implementation of ``nn.MultiheadAttention``

    Parameters:
    -----------
    config : BalmMoEConfig
        Model configuration class with all the parameters of the model.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_experts: int,
        expert_capacity: int,
        num_shared_experts: int = 0,
        top_k: int = 1,
        expert_activation: str = "gelu",
        expert_ffn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_batch_first: bool = True,
        layer_norm_eps: float = 1e-5,
        router_dtype: str = "float32",
        router_bias: bool = False,
        router_jitter: float = 0.0,
        router_ignore_padding_tokens: bool = True,
        router_class: nn.Module = TopKRouter,
        expert_class: nn.Module = Expert,
        # config: BalmMoEConfig,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.expert_ffn_dropout = expert_ffn_dropout
        self.layer_norm_eps = layer_norm_eps

        # can't use rotary embeddings with nn.MultiheadAttention
        # see: https://discuss.pytorch.org/t/is-there-a-way-to-implement-rope-around-nn-multiheadattention-somehow/175051
        # it is possible to use rotary embeddings with F.scaled_dot_product_attention,
        # but it's not clear that it's worth the effort
        # see: https://github.com/pytorch/pytorch/issues/97899 for an example
        # self.use_rotary_embeddings = use_rotary_embeddings

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.attention_dropout,
            batch_first=attention_batch_first,
        )

        self.mlp = SparseMLP(
            embed_dim=self.embed_dim,
            ffn_dim=self.ffn_dim,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            expert_capacity=expert_capacity,
            expert_activation=expert_activation,
            expert_ffn_dropout=expert_ffn_dropout,
            router_dtype=router_dtype,
            router_bias=router_bias,
            router_jitter=router_jitter,
            router_ignore_padding_tokens=router_ignore_padding_tokens,
            router_class=router_class,
            expert_class=expert_class,
        )
        self.ff_dropout = nn.Dropout(self.ffn_dropout)

        self.norm1 = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        output_router_logits: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        """
        Process the input hidden states.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        attn_mask : torch.Tensor, optional
            Attention mask of shape (batch_size * num_heads, sequence_length, sequence_length). The default is None.

        key_padding_mask : torch.Tensor, optional
            Mask of shape (batch_size, sequence_length). The default is None.

        need_weights : bool, optional
            Whether to return attention weights. The default is False.

            .. note::
                if `need_weights` is ``True``, the output will be a tuple of (x, attn). Also,
                nn.MultiHeadAttention will not be able to use the optimized torch implementation
                of ``scaled_dot_product_attention``. See `here`_ for more details.

        output_router_logits : bool, optional
            Whether to output router logits. The default is True.

        Returns:
        --------
        x : torch.Tensor or Tuple

            Output tensor of shape (batch_size, sequence_length, embed_dim). If `need_weights`, is ``True``,
            output is a tuple of (x, attn). If `output_router_logits` is ``True``, the output will be a tuple
            of (x, router_logits) or (x, attn, router_logts) depending on the value of `need_weights`.


        .. _here:
            https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
        """
        # attention
        residual = x
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attention_mask,
        )
        if need_weights:
            x, attn = x
        x = residual + x
        x = self.norm1(x)

        # sparse feedforward
        residual = x
        x, router_tuple = self.mlp(x)  # router_tuple is (router_logits, expert_index)
        x = self.ff_dropout(x)
        x = self.norm2(residual + x)
        if output_router_logits and router_tuple is not None:
            if need_weights:
                return (x, attn, router_tuple)
            return (x, router_tuple)
        if need_weights:
            return (x, attn)
        return x


class SparseRoformerLayer(nn.Module):
    """
    Sparse Roformer layer.


    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_experts: int,
        max_len: int,
        expert_capacity: int,
        expert_activation: str = "gelu",
        expert_ffn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_batch_first: bool = True,
        layer_norm_eps: float = 1e-5,
        router_dtype: str = "float32",
        router_bias: bool = False,
        router_jitter: float = 0.0,
        router_ignore_padding_tokens: bool = True,
        router_class: nn.Module = TopKRouter,
        expert_class: nn.Module = Expert,
    ):
        super().__init__()
        self.rotary_embedding = RotaryPositionalEmbeddings(embed_dim, max_len)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            # batch_first=attention_batch_first,
        )

        self.mlp = SparseMLP(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            expert_capacity=expert_capacity,
            expert_activation=expert_activation,
            expert_ffn_dropout=expert_ffn_dropout,
            router_dtype=router_dtype,
            router_bias=router_bias,
            router_jitter=router_jitter,
            router_ignore_padding_tokens=router_ignore_padding_tokens,
            router_class=router_class,
            expert_class=expert_class,
        )
        self.ff_dropout = nn.Dropout(ffn_dropout)

        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        output_router_logits: bool = True,
    ):
        # pre-norm
        query_norm = self.norm1(query)
        key_norm = self.norm1(key)
        value_norm = self.norm1(value)

        # rotary embeddings
        query_rot = self.rotary_embedding(query_norm)
        key_rot = self.rotary_embedding(key_norm)
        value_rot = self.rotary_embedding(value_norm)

        # attention
        x, _ = self.attention(
            query_rot,
            key_rot,
            value_rot,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        # x = query + self.dropout(x)
        x = query + x
        if need_weights:
            x, attn = x

        # pre-norm
        residual = x
        x = self.norm2(x)

        # sparse feedforward
        x, router_tuple = self.mlp(x)
        x = residual + self.ff_dropout(x)

        if output_router_logits and router_tuple is not None:
            if need_weights:
                return (x, attn, router_tuple)
            return (x, router_tuple)
        if need_weights:
            return (x, attn)
        return x
