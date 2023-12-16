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


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BALMMoEConfig
from .loss import router_z_loss, load_balancing_loss_func



class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        add_bias_kv: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.drop_p = dropout_p

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
            dropout=self.drop_p,
        )

        self.ff_linear_1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.ff_linear_2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.ff_dropout = nn.Dropout(self.drop_p)
        self.ff_activation = nn.GELU()

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)


    def forward(
        self, x, attn_mask=None, key_padding_mask=None, need_weights=True
    ):
        # attention
        residual = x
        x = self.norm1(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )
        x = residual + x

        # feedforward
        residual = x
        x = self.norm2(x)
        x = self.ff_linear_2(self.ff_activation(self.ff_linear_1(x)))
        x = self.ff_dropout(x)
        x = residual + x

        return x, attn





# ---------------------------
#     SWITCH TRANSFORMERS
# ---------------------------

class Top1Router(nn.Module):
    """
    Router that allows tokens to choose their own top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their expert of choice until the expert's `expert_capacity` is reached. **There is no guarantee that 
    each token is processed by an expert**, or that each expert receives at least one token.
    """

    def __init__(self, config: BALMMoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.dtype = getattr(torch, config.router_dtype)
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias, dtype=self.dtype)
        self.jitter = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens

    def _compute_router_probabilities(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes router probabilities from input hidden states.

        Args:
            x (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            router_probabilities (`torch.Tensor`):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor`):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        # float32 is used to ensure stability. See the discussion of "selective precision" in
        # https://arxiv.org/abs/2101.03961.
        # we also store the previous dtype to cast the output back to the original dtype
        self.input_dtype = x.dtype
        x = x.to(self.dtype)
        if self.jitter > 0:
            x *= torch.empty_like(x).uniform_(1.0 - self.jitter, 1.0 + self.jitter)

        # shape: [batch_size, sequence_length, num_experts]
        logits = self.classifier(x)

        # apply softmax and cast back to the original `dtype`
        probabilities = F.softmax(logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return probabilities, logits

    def forward(self, x: torch.Tensor) -> Tuple:
        r"""
        Generic forward function for every Router class. Each Router expects to have the same input hidden states
        (`x`) corresponding to the hidden states for each token, the `expert_capacity` corresponding to the
        number of tokens the Router can send to each expert.

        Each Router works as follows: from the hidden states for each token, it gets the `router_probs` and
        `router_logits`. This will assign for each token, the raw probability that it should be assigned
        to each expert.

        Args:
            x (`torch.Tensor`) :
                (batch_size, sequence_length, hidden_dim) inputs to send to experts.
        Returns:
            Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`] Tuple containing the expert index, the router probs
            and the router logits. The router probabilities and logits are required to compute the loss.
        """
        router_probs, router_logits = self._compute_router_probabilities(x)
        expert_indices = torch.argmax(router_probs, dim=-1)
        expert_indices = F.one_hot(expert_indices, num_classes=self.num_experts)

        # sum over each sequence
        token_priority = torch.cumsum(expert_indices, dim=-2)
        # mask tokens if their desired expert is above capacity
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_indices = expert_indices * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_indices, router_probs, router_logits



# This is from the huggingface implementation of Switch Transformers
# https://github.com/huggingface/transformers/blob/c48787f347bd604f656c2cfff730e029c8f8c1fe/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L223
# Not sure if it's needed -- the main difference between this and nn.LayerNorm is that this one
# doesn't subtract the mean from the input (and doesn't have bias, but we can do that in nn.LayerNorm)

# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->SwitchTransformers
class SwitchTransformersLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the SwitchTransformers style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # SwitchTransformers uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states



# Copied from transformers.models.t5.modeling_t5.T5DenseActDense with T5->SwitchTransformers
class Expert(nn.Module):
    def __init__(self, config: BALMMoEConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_rate)
        # in the huggingface implementation, relu is used by default as the activation function
        # self.activation = ACT2FN[config.dense_act_fn]
        self.activation = nn.GELU()

    def forward(self, x):
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
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: BALMMoEConfig, expert_class: nn.Module = Expert):
        super().__init__()
        # Step 1: Get the correct router according to its class
        self.router = Top1Router(config)

        # Step 2: Get the experts
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)

    def forward(self, x):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        # get the router mask, probabilities, and logits
        # mask shape: [batch_size, sequence_length, num_experts]
        router_mask, router_probs, router_logits = self.router(x)
        expert_index = torch.argmax(router_mask, dim=-1)

        # The routers introduced might not always map all the tokens to a router, since
        # the desired expert might be above capacity. The hidden states of those tokens
        # will be unchanged from one layer to another. That is why the hidden states are 
        # cloned before updating only the tokens that have been successfully routed to an expert.
        next_states = x.clone()
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert(x[token_indices]).to(next_states.dtype)

        x = router_probs * next_states
        return x, (router_logits, expert_index)


class FFLayerMoE(nn.Module):
    """
    Switch Transformers Feed Forward layer module. This is a wrapper around the Mixture of Experts module.

    Parameters:
        config : ([`SwitchTransformersConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        is_sparse (`bool`):
            Whether the MLP layer is a `Sparse` layer (contains a Mixture of Experts) or not
    """

    def __init__(self, config: BALMMoEConfig):
        super().__init__()
        self.mlp = SparseMLP(config)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, output_router_logits):
        residual = x
        x, router_tuple = self.mlp(x)
        x = self.dropout(x)
        x = self.layer_norm(residual + x)
        if output_router_logits and router_tuple is not None:
            return (x, router_tuple)
        return x



        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        forwarded_states, router_tuple = forwarded_states
        output = hidden_states + self.dropout(forwarded_states)
        if output_router_logits and router_tuple is not None:
            output = (output, router_tuple)
        return output





class TransformerLayerMoE(nn.Module):
    """Transformer layer block, with Mixture of Experts."""

    def __init__(
        self,
        config: BALMMoEConfig,
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
            dropout=self.attn.dropout_rate,
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
        need_weights: bool = True, 
        output_router_logits: bool = True,
    ):
        # attention
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )
        x = residual + x
        x = self.norm1(x)

        # sparse feedforward
        residual = x
        x, router_tuple = self.mlp(x)
        x = self.dropout(x)
        x = self.layer_norm(residual + x)
        if output_router_logits and router_tuple is not None:
            return (x, router_tuple)
        return x


        # feedforward
        residual = x
        x = self.norm2(x)
        x = self.ff_linear_2(self.ff_activation(self.ff_linear_1(x)))
        x = self.ff_dropout(x)
        x = residual + x

        return x, attn





class SwitchTransformersBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, is_sparse=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.is_sparse = is_sparse
        self.layer = nn.ModuleList()
        self.layer.append(
            SwitchTransformersLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        )
        if self.is_decoder:
            self.layer.append(SwitchTransformersLayerCrossAttention(config))

        self.layer.append(SwitchTransformersLayerFF(config, is_sparse=self.is_sparse))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        output_router_logits=True,
        return_dict=True,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, output_router_logits)

        if isinstance(hidden_states, tuple):
            hidden_states, router_tuple = hidden_states
        else:
            router_tuple = (torch.zeros((1,), device=hidden_states.device, dtype=torch.int64),)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs + (router_tuple,)
        else:
            outputs = outputs + attention_outputs + (router_tuple,)

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights), (router_tuple)
























class RoBERTaLMHead(nn.Module):
    """
    Head for masked language modeling.
    """

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim, bias=False)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = nn.GELU(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ContactPredictionHead(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        prepend_bos: bool,
        append_eos: bool,
        bias=True,
        eos_idx: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        if append_eos and eos_idx is None:
            raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias)
        self.activation = nn.Sigmoid()

    def forward(self, tokens, attentions):
        # remove eos token attentions
        if self.append_eos:
            eos_mask = tokens.ne(self.eos_idx).to(attentions)
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: B x C x T x T
        attentions = attentions.to(
            self.regression.weight.device
        )  # attentions always float32, may need to convert to float16
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        return self.activation(self.regression(attentions).squeeze(3))

def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


