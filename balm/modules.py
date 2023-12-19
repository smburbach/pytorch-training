#!/usr/bin/python
# filename: modules.py

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


from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BalmMoEConfig


class TransformerLayer(nn.Module):
    """
    BALM transformer layer. Approximately follows the ESM-2 implementation,
    but differs in a couple ways:
        - includes (optional) dropout
        - we don't use rotary embeddings, which aren't (yet?) compatible with
          torch's optimized implementation of nn.MultiheadAttention

    Parameters:
    -----------
    embed_dim : int
        Embedding dimension.

    ffn_embed_dim : int
        Feed-forward network embedding dimension. Typically 4x the embedding dimension.

    attention_heads : int
        Number of attention heads.

    add_bias_kv : bool, optional
        Whether to add bias to the key and value projections. The default is ``True``.

    dropout_rate : float, optional
        Dropout rate. The default is ``0.0``.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        add_bias_kv: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate

        # can't use rotary embeddings with nn.MultiheadAttention
        # see: https://discuss.pytorch.org/t/is-there-a-way-to-implement-rope-around-nn-multiheadattention-somehow/175051
        # it is possible to use rotary embeddings with F.scaled_dot_product_attention,
        # but it's not clear that it's worth the effort
        # see: https://github.com/pytorch/pytorch/issues/97899 for an example
        # self.use_rotary_embeddings = use_rotary_embeddings

        self.self_attn = nn.MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            dropout=self.dropout_rate,
        )

        self.ff_linear_1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.ff_linear_2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.ff_dropout = nn.Dropout(self.dropout_rate)
        self.ff_activation = nn.GELU()

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        """

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        attn_mask : torch.Tensor, optional
            Attention mask of shape (batch_size, sequence_length, sequence_length). The default is None.

        key_padding_mask : torch.Tensor, optional
            Mask of shape (batch_size, sequence_length). The default is None.

        need_weights : bool, optional
            Whether to return attention weights. The default is False.

            .. NOTE: if `need_weights` is ``True``, the output will be a tuple of (x, attn). Also,
            nn.MultiHeadAttention will not be able to use the optimized torch implementation
            of ``scaled_dot_product_attention``. See `here`_ for more details.

        Returns:
        --------
        x : torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Output tensor of shape (batch_size, sequence_length, embed_dim). If `need_weights` is
            ``True``, the output will be a tuple of (x, attn).

        .. _here:
            https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
        """
        # attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )
        if need_weights:
            x, attn = x
        x = residual + x

        # feedforward
        residual = x
        x = self.norm2(x)
        x = self.ff_linear_2(self.ff_activation(self.ff_linear_1(x)))
        x = self.ff_dropout(x)
        x = residual + x

        if need_weights:
            return x, attn
        return x


# ---------------------------
#        BALM MoE
# ---------------------------


class Router(nn.Module):
    """
    Default router for BALM MoE models.

    This router uses the mechanism introduced by `Switch Transformers`_, in which
    tokens select a single expert. Tokens are routed to their expert of choice until the
    expert's `expert_capacity` is reached. **There is no guarantee that each token will be
    processed by an expert**, or that every expert will receive a token.

    If tokens are routed to an expert above capacity, they are not processed by any expert
    and their hidden states are passed to the subsequent layer unchanged.


    Parameters:
    -----------
    embed_dim : int
        Embedding dimension.

    num_experts : int
        Number of experts.

    expert_capacity : int
        Maximum number of tokens that can be routed to each expert.

    dtype : str, optional
        Data type to use for router probabilities. The default is "float32".

    bias : bool, optional
        Whether to add bias to the router classifier. The default is ``False``.

    jitter : float, optional
        Amount of jitter to add to the router probabilities. The default is ``0.0``.

    ignore_padding_tokens : bool, optional
        Whether to ignore padding tokens when computing router probabilities.
        The default is ``True``.


    .. _Switch Transformers:
        https://arxiv.org/abs/2101.03961
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        expert_capacity: int,
        dtype: str = "float32",
        bias: bool = False,
        jitter: float = 0.0,
        ignore_padding_tokens: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.dtype = getattr(torch, dtype)
        self.classifier = nn.Linear(
            embed_dim,
            self.num_experts,
            bias=bias,
            dtype=self.dtype,
        )
        self.jitter = jitter
        self.ignore_padding_tokens = ignore_padding_tokens

    def _compute_router_probabilities(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes router probabilities from input hidden states.

        Parameters:
        -----------
        x : torch.Tensor
            Tensor of shape (batch_size, sequence_length, hidden_dim) from which
            router probabilities are computed.

        Returns:
        --------
        router_probabilities : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) corresponding to
            the probabilities for each token and expert. Used for routing tokens to experts.

        router_logits : torch.Tensor
            Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding
            to raw router logits. This is used for computing router z-loss.
        """
        # float32 is used to ensure stability. See the discussion of "selective precision" in
        # https://arxiv.org/abs/2101.03961.
        # we also store the input dtype so we can cast the output back to the original dtype
        self.input_dtype = x.dtype
        x = x.to(self.dtype)
        if self.jitter > 0:
            x *= torch.empty_like(x).uniform_(1.0 - self.jitter, 1.0 + self.jitter)

        # shape: [batch_size, sequence_length, num_experts]
        logits = self.classifier(x)

        # apply softmax and cast back to the original dtype
        probabilities = F.softmax(logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return probabilities, logits

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
        --------
        expert_indices : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) indicating
            which expert the token should be routed to.

        router_probabilities : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) containing
            the router probabilities.

        router_logits : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) containing
            the router logits.
        """
        router_probs, router_logits = self._compute_router_probabilities(x)
        expert_indices = torch.argmax(router_probs, dim=-1)
        expert_indices = F.one_hot(expert_indices, num_classes=self.num_experts)

        # mask tokens if their desired expert is above capacity
        token_priority = torch.cumsum(expert_indices, dim=-2)
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_indices = expert_indices * expert_capacity_mask

        # get the probability of the top-choice expert for each token
        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)

        return expert_indices, router_probs, router_logits


# This is from the huggingface implementation of Switch Transformers
# https://github.com/huggingface/transformers/blob/c48787f347bd604f656c2cfff730e029c8f8c1fe/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L223
# Not sure if it's needed -- the main difference between this and nn.LayerNorm is that this one
# doesn't subtract the mean from the input (and doesn't have bias, but we can do that in nn.LayerNorm)

# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->SwitchTransformers

# class SwitchTransformersLayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         Construct a layernorm module in the SwitchTransformers style. No bias and no subtraction of mean.
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         # SwitchTransformers uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
#         # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
#         # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
#         # half-precision inputs is done in fp32

#         variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

#         # convert into half-precision if necessary
#         if self.weight.dtype in [torch.float16, torch.bfloat16]:
#             hidden_states = hidden_states.to(self.weight.dtype)

#         return self.weight * hidden_states


# Copied from transformers.models.t5.modeling_t5.T5DenseActDense with T5->SwitchTransformers
class Expert(nn.Module):
    """
    Implementation of the Switch Transformers Expert module.

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
        ffn_embed_dim: int,
        dropout_rate: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.wi = nn.Linear(embed_dim, ffn_embed_dim, bias=False)
        self.wo = nn.Linear(ffn_embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        # in the huggingface implementation, relu is used by default as the activation function
        # self.activation = ACT2FN[config.dense_act_fn]
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
        Router class to use. The default is ``Router``.

    expert_class : nn.Module, optional
        Expert class to use. The default is ``Expert``.

    """

    def __init__(
        self,
        config: BalmMoEConfig,
        router_class: nn.Module = Router,
        expert_class: nn.Module = Expert,
    ):
        super().__init__()
        self.router = router_class(
            embed_dim=config.embed_dim,
            num_experts=config.num_experts,
            expert_capacity=config.expert_capacity,
            dtype=config.router_dtype,
            bias=config.router_bias,
            jitter=config.router_jitter,
            ignore_padding_tokens=config.router_ignore_padding_tokens,
        )
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(
                embed_dim=config.embed_dim,
                ffn_embed_dim=config.ffn_embed_dim,
                dropout_rate=config.hidden_dropout_rate,
                activation=config.expert_activation,
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
        # mask shape: [batch_size, sequence_length, num_experts]
        router_mask, router_probs, router_logits = self.router(x)
        expert_index = torch.argmax(router_mask, dim=-1)

        # The router might not always map all the tokens to a n expert, since
        # the top-choice expert might be above capacity. The hidden states of those tokens
        # will be unchanged from one layer to another. That is why the hidden states are
        # cloned before updating only the tokens that have been successfully routed to an expert.
        next_states = x.clone()
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert(x[token_indices]).to(next_states.dtype)

        x = router_probs * next_states
        return x, (router_logits, expert_index)


# class FFLayerMoE(nn.Module):
#     """
#     Feedforward layer with Mixture of Experts.

#     Parameters:
#     -----------
#     config : BalmMoEConfig
#         Model configuration class.

#     """

#     def __init__(self, config: BalmMoEConfig):
#         super().__init__()
#         self.mlp = SparseMLP(config)
#         self.layer_norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_rate)

#     def forward(
#         self, x: torch.Tensor, output_router_logits: bool = True
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
#         """
#         Process the input hidden states.

#         Parameters:
#         -----------
#         x : torch.Tensor
#             Input tensor of shape (batch_size, sequence_length, embed_dim).

#         output_router_logits : bool, optional
#             Whether to output router logits. The default is True.

#         Returns:
#         --------
#         x : torch.Tensor or Tuple[torch.Tensor, Tuple]
#             Output tensor of shape (batch_size, sequence_length, embed_dim). If `output_router_logits` is
#             ``True``, the output will be a tuple of (x, router_logits).
#         """
#         residual = x
#         # router_tuple is (router_logits, expert_index)
#         x, router_tuple = self.mlp(x)
#         x = self.dropout(x)
#         x = self.layer_norm(residual + x)
#         if output_router_logits and router_tuple is not None:
#             return (x, router_tuple)
#         return x


class MoETransformerLayer(nn.Module):
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
        config: BalmMoEConfig,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.ffn_embed_dim = config.d_ff
        self.attention_heads = config.num_heads
        self.attn_dropout_rate = config.attn_dropout_rate
        self.hidden_dropout_rate = config.hidden_dropout_rate
        self.layer_norm_eps = config.layer_norm_eps

        # can't use rotary embeddings with nn.MultiheadAttention
        # see: https://discuss.pytorch.org/t/is-there-a-way-to-implement-rope-around-nn-multiheadattention-somehow/175051
        # it is possible to use rotary embeddings with F.scaled_dot_product_attention,
        # but it's not clear that it's worth the effort
        # see: https://github.com/pytorch/pytorch/issues/97899 for an example
        # self.use_rotary_embeddings = use_rotary_embeddings

        self.self_attn = nn.MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            dropout=self.attn_dropout_rate,
        )

        self.sparse_mlp = SparseMLP(config)
        self.ff_dropout = nn.Dropout(self.hidden_dropout_rate)

        self.norm1 = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
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
            Attention mask of shape (batch_size, sequence_length, sequence_length). The default is None.

        key_padding_mask : torch.Tensor, optional
            Mask of shape (batch_size, sequence_length). The default is None.

        need_weights : bool, optional
            Whether to return attention weights. The default is False.

            .. NOTE: if `need_weights` is ``True``, the output will be a tuple of (x, attn). Also,
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
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )
        if need_weights:
            x, attn = x
        x = residual + x
        x = self.norm1(x)

        # sparse feedforward
        residual = x
        x, router_tuple = self.mlp(x)  # router_tuple is (router_logits, expert_index)
        x = self.dropout(x)
        x = self.norm2(residual + x)
        if output_router_logits and router_tuple is not None:
            if need_weights:
                return (x, attn, router_tuple)
            return (x, router_tuple)
        if need_weights:
            return x, attn
        return x


# class MoETransformerStack(nn.Module):
#     """ """

#     def __init__(self, config: BalmMoEConfig):
#         # embedding
#         self.embed_scale = 1.0 if config.scale_embedding else math.sqrt(config.d_model)
#         self.embed_tokens = nn.Embedding(
#             config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
#         )
#         # build the block
#         self.layers = nn.ModuleList(
#             [MoETransformerLayer(config) for _ in range(config.num_layers)]
#         )

#         self.layers = nn.ModuleList()
#         for _ in range(config.num_layers):
#             self.stack.append(MoETransformerLayer(config))
#         self.final_layer_norm = nn.LayerNorm(
#             config.embed_dim, eps=config.layer_norm_eps
#         )
#         self.dropout = nn.Dropout(config.hidden_dropout_rate)
#         # initialize weights
#         self.post_init()


# class SwitchTransformersStack(SwitchTransformersPreTrainedModel):
#     def __init__(self, config, embed_tokens=None):
#         super().__init__(config)

#         self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

#         if embed_tokens is not None:
#             self.embed_tokens.weight = embed_tokens.weight

#         self.is_decoder = config.is_decoder

#         sparse_step = (
#             config.decoder_sparse_step
#             if self.is_decoder
#             else config.encoder_sparse_step
#         )
#         config.num_layers = (
#             config.num_decoder_layers if self.is_decoder else config.num_layers
#         )
#         self.block = nn.ModuleList()
#         for i in range(config.num_layers):
#             is_sparse = (i % sparse_step == 1) if sparse_step > 0 else False

#             self.block.append(
#                 SwitchTransformersBlock(
#                     config,
#                     has_relative_attention_bias=bool(i == 0),
#                     is_sparse=is_sparse,
#                 )
#             )

#         self.final_layer_norm = SwitchTransformersLayerNorm(
#             config.d_model, eps=config.layer_norm_epsilon
#         )
#         self.dropout = nn.Dropout(config.dropout_rate)

#         # Initialize weights and apply final processing
#         self.post_init()

#         self.device_map = None
#         self.gradient_checkpointing = False

#     def get_input_embeddings(self):
#         return self.embed_tokens

#     def set_input_embeddings(self, new_embeddings):
#         self.embed_tokens = new_embeddings

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         inputs_embeds=None,
#         head_mask=None,
#         cross_attn_head_mask=None,
#         past_key_values=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         output_router_logits=True,
#         return_dict=None,
#     ):
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         output_attentions = (
#             output_attentions
#             if output_attentions is not None
#             else self.config.output_attentions
#         )
#         output_hidden_states = (
#             output_hidden_states
#             if output_hidden_states is not None
#             else self.config.output_hidden_states
#         )
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )

#         if input_ids is not None and inputs_embeds is not None:
#             err_msg_prefix = "decoder_" if self.is_decoder else ""
#             raise ValueError(
#                 f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
#             )
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#             input_ids = input_ids.view(-1, input_shape[-1])
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             err_msg_prefix = "decoder_" if self.is_decoder else ""
#             raise ValueError(
#                 f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
#             )

#         if inputs_embeds is None:
#             if self.embed_tokens is None:
#                 raise ValueError(
#                     "You have to initialize the model with valid token embeddings"
#                 )
#             inputs_embeds = self.embed_tokens(input_ids)

#         batch_size, seq_length = input_shape

#         # required mask seq length can be calculated via length of past
#         mask_seq_length = (
#             past_key_values[0][0].shape[2] + seq_length
#             if past_key_values is not None
#             else seq_length
#         )

#         if use_cache is True:
#             if not self.is_decoder:
#                 raise ValueError(
#                     f"`use_cache` can only be set to `True` if {self} is used as a decoder"
#                 )

#         if attention_mask is None:
#             attention_mask = torch.ones(
#                 batch_size, mask_seq_length, device=inputs_embeds.device
#             )
#         if (
#             self.is_decoder
#             and encoder_attention_mask is None
#             and encoder_hidden_states is not None
#         ):
#             encoder_seq_length = encoder_hidden_states.shape[1]
#             encoder_attention_mask = torch.ones(
#                 batch_size,
#                 encoder_seq_length,
#                 device=inputs_embeds.device,
#                 dtype=torch.long,
#             )

#         # initialize past_key_values with `None` if past does not exist
#         if past_key_values is None:
#             past_key_values = [None] * len(self.block)

#         # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#         # ourselves in which case we just need to make it broadcastable to all heads.
#         extended_attention_mask = self.get_extended_attention_mask(
#             attention_mask, input_shape
#         )

#         # If a 2D or 3D attention mask is provided for the cross-attention
#         # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
#         if self.is_decoder and encoder_hidden_states is not None:
#             (
#                 encoder_batch_size,
#                 encoder_sequence_length,
#                 _,
#             ) = encoder_hidden_states.size()
#             encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#             if encoder_attention_mask is None:
#                 encoder_attention_mask = torch.ones(
#                     encoder_hidden_shape, device=inputs_embeds.device
#                 )
#             encoder_extended_attention_mask = self.invert_attention_mask(
#                 encoder_attention_mask
#             )
#         else:
#             encoder_extended_attention_mask = None

#         if self.gradient_checkpointing and self.training:
#             if use_cache:
#                 logger.warning_once(
#                     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#                 )
#                 use_cache = False

#         # Prepare head mask if needed
#         head_mask = self.get_head_mask(head_mask, self.config.num_layers)
#         cross_attn_head_mask = self.get_head_mask(
#             cross_attn_head_mask, self.config.num_layers
#         )
#         present_key_value_states = () if use_cache else None
#         all_hidden_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None
#         all_router_probs = () if output_router_logits else None
#         all_cross_attentions = () if (output_attentions and self.is_decoder) else None
#         position_bias = None
#         encoder_decoder_position_bias = None

#         hidden_states = self.dropout(inputs_embeds)

#         for i, (layer_module, past_key_value) in enumerate(
#             zip(self.block, past_key_values)
#         ):
#             layer_head_mask = head_mask[i]
#             cross_attn_layer_head_mask = cross_attn_head_mask[i]

#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     layer_module.forward,
#                     hidden_states,
#                     extended_attention_mask,
#                     position_bias,
#                     encoder_hidden_states,
#                     encoder_extended_attention_mask,
#                     encoder_decoder_position_bias,
#                     layer_head_mask,
#                     cross_attn_layer_head_mask,
#                     None,  # past_key_value is always None with gradient checkpointing
#                     use_cache,
#                     output_attentions,
#                 )
#             else:
#                 layer_outputs = layer_module(
#                     hidden_states,
#                     attention_mask=extended_attention_mask,
#                     position_bias=position_bias,
#                     encoder_hidden_states=encoder_hidden_states,
#                     encoder_attention_mask=encoder_extended_attention_mask,
#                     encoder_decoder_position_bias=encoder_decoder_position_bias,
#                     layer_head_mask=layer_head_mask,
#                     cross_attn_layer_head_mask=cross_attn_layer_head_mask,
#                     past_key_value=past_key_value,
#                     use_cache=use_cache,
#                     output_attentions=output_attentions,
#                     output_router_logits=output_router_logits,
#                 )

#             router_probs = layer_outputs[-1]
#             layer_outputs = layer_outputs[:-1]

#             # layer_outputs is a tuple with:
#             # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
#             if use_cache is False:
#                 layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

#             hidden_states, present_key_value_state = layer_outputs[:2]

#             # We share the position biases between the layers - the first layer store them
#             # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
#             # (cross-attention position bias), (cross-attention weights)
#             position_bias = layer_outputs[2]
#             if self.is_decoder and encoder_hidden_states is not None:
#                 encoder_decoder_position_bias = layer_outputs[
#                     4 if output_attentions else 3
#                 ]
#             # append next layer key value states
#             if use_cache:
#                 present_key_value_states = present_key_value_states + (
#                     present_key_value_state,
#                 )

#             if output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[3],)
#                 if self.is_decoder:
#                     all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

#             if output_router_logits:
#                 all_router_probs = all_router_probs + (router_probs,)

#         hidden_states = self.final_layer_norm(hidden_states)
#         hidden_states = self.dropout(hidden_states)

#         # Add last layer
#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [
#                     hidden_states,
#                     present_key_value_states,
#                     all_hidden_states,
#                     all_attentions,
#                     all_cross_attentions,
#                     all_router_probs,
#                 ]
#                 if v is not None
#             )
#         return MoEModelOutputWithPastAndCrossAttentions(
#             last_hidden_state=hidden_states,
#             past_key_values=present_key_value_states,
#             hidden_states=all_hidden_states,
#             attentions=all_attentions,
#             cross_attentions=all_cross_attentions,
#             router_probs=all_router_probs,
#         )


class MaskedLMHead(nn.Module):
    """
    Masked language modeling head for BALM models. Implements the same LM head as `ESM-2`_.

    Parameters:
    -----------
    embed_dim : int
        Embedding dimension.

    output_dim : int
        Output dimension.

    weight : torch.Tensor
        Embedding weight matrix.


    .. ESM-2:
        https://www.biorxiv.org/content/10.1101/622803v2
    """

    def __init__(self, embed_dim: int, output_dim: int, weight: torch.Tensor):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Process features into outputs.

        Parameters:
        -----------
        features : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        x = self.dense(features)
        x = nn.GELU(x)
        x = self.norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


# class ContactPredictionHead(nn.Module):
#     """
#     Performs symmetrization, apc, and computes a logistic regression on the output features
#     """

#     def __init__(
#         self,
#         in_features: int,
#         prepend_bos: bool,
#         append_eos: bool,
#         bias=True,
#         eos_idx: Optional[int] = None,
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.prepend_bos = prepend_bos
#         self.append_eos = append_eos
#         if append_eos and eos_idx is None:
#             raise ValueError(
#                 "Using an alphabet with eos token, but no eos token was passed in."
#             )
#         self.eos_idx = eos_idx
#         self.regression = nn.Linear(in_features, 1, bias)
#         self.activation = nn.Sigmoid()

#     def forward(self, tokens, attentions):
#         # remove eos token attentions
#         if self.append_eos:
#             eos_mask = tokens.ne(self.eos_idx).to(attentions)
#             eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
#             attentions = attentions * eos_mask[:, None, None, :, :]
#             attentions = attentions[..., :-1, :-1]
#         # remove cls token attentions
#         if self.prepend_bos:
#             attentions = attentions[..., 1:, 1:]
#         batch_size, layers, heads, seqlen, _ = attentions.size()
#         attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

#         # features: B x C x T x T
#         attentions = attentions.to(
#             self.regression.weight.device
#         )  # attentions always float32, may need to convert to float16
#         attentions = apc(symmetrize(attentions))
#         attentions = attentions.permute(0, 2, 3, 1)
#         return self.activation(self.regression(attentions).squeeze(3))


# def symmetrize(x):
#     "Make layer symmetric in final two dimensions, used for contact prediction."
#     return x + x.transpose(-1, -2)


# def apc(x):
#     "Perform average product correct, used for contact prediction."
#     a1 = x.sum(-1, keepdims=True)
#     a2 = x.sum(-2, keepdims=True)
#     a12 = x.sum((-1, -2), keepdims=True)

#     avg = a1 * a2
#     avg.div_(a12)  # in-place to reduce memory
#     normalized = x - avg
#     return normalized
