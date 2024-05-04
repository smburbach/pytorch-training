#!/usr/bin/python
# filename: balm.py

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

import torch
import torch.nn as nn

from ..config import BalmConfig
from ..modules import BalmLMHead, MaskedLMOutput, RoformerLayer
from .base import BalmBase


class BalmModel(BalmBase):
    config_cls = BalmConfig

    def __init__(
        self,
        config: BalmConfig,
        # embed_dim: int = 512,
        # ffn_dim: int = 2048,
        # num_layers: int = 6,
        # num_heads: int = 8,
        # vocab_size: int = 33,
        # max_length: int = 320,
        # dropout: float = 0.0,
        # attention_dropout: float = 0.1,
        # token_embedding_dropout: float = 0.0,
        # layer_norm_eps: float = 1e-5,
        # padding_idx: int = 0,
    ):
        """
        BALM model, with rotary embeddings, pre-norm, and SwiGLU activations.

        Parameters
        ----------
        embed_dim : int, default = 512
            The input embedding dimension.

        ffn_dim : int, default = 2048
            The dimension of the feedforward network.

        num_layers : int, default = 6
            The number of layers.

        num_heads : int, default = 8
            The number of attention heads.

        vocab_size : int, default = 33
            The size of the vocabulary.

        max_length : int, default = 320
            The maximum sequence length.

        dropout : float, default = 0.0
            The dropout probability.

        attention_dropout : float, default = 0.1
            The dropout probability for the attention weights.

        token_embedding_dropout : float, default = 0.0
            The dropout probability for the token embeddings.

        layer_norm_eps : float, default = 1e-5
            The epsilon value for the layer normalization.

        """
        super().__init__()
        # self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # self.embedding_dropout = nn.Dropout(token_embedding_dropout)
        # self.layers = nn.ModuleList(
        #     [
        #         RoformerLayer(
        #             embed_dim,
        #             ffn_dim,
        #             num_heads,
        #             max_length,
        #             dropout=dropout,
        #             attention_dropout=attention_dropout,
        #             layer_norm_eps=layer_norm_eps,
        #         )
        #         for _ in range(num_layers)
        #     ]
        # )
        # self.final_layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.config = config
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.embed_dim,
            padding_idx=self.config.padding_idx,
        )
        self.embedding_dropout = nn.Dropout(self.config.token_embedding_dropout)
        self.layers = nn.ModuleList(
            [
                RoformerLayer(
                    self.config.embed_dim,
                    self.config.ffn_dim,
                    self.config.num_heads,
                    self.config.max_length,
                    dropout=self.config.dropout,
                    attention_dropout=self.config.attention_dropout,
                    layer_norm_eps=self.config.layer_norm_eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(
            self.config.embed_dim, eps=self.config.layer_norm_eps
        )

    # @property
    # def num_parameters(self):
    #     return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------

        x : torch.Tensor
            The input tensor. Expected shape is (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            The output tensor. The shape is (batch_size, seq_len, embed_dim).
        """
        x = self.embed_tokens(x)
        x = self.embedding_dropout(x)
        for layer in self.layers:
            x = layer(
                x,
                x,
                x,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
            )
            if need_weights:
                x, attn = x
        x = self.final_layer_norm(x)
        if need_weights:
            return x, attn
        return x


class BalmForMaskedLM(BalmBase):
    config_cls = BalmConfig

    def __init__(
        self,
        config: BalmConfig,
        # embed_dim: int = 512,
        # ffn_dim: int = 2048,
        # num_layers: int = 6,
        # num_heads: int = 8,
        # vocab_size: int = 33,
        # max_length: int = 320,
        # dropout: float = 0.0,
        # attention_dropout: float = 0.1,
        # token_embedding_dropout: float = 0.0,
        # layer_norm_eps: float = 1e-5,
        # padding_idx: int = 0,
    ):
        """
        BALM model for masked language modeling. Uses the base BALM model with rotary embeddings, pre-norm,
        and SwiGLU activations, and adds a language modeling head.

        Parameters
        ----------
        embed_dim : int, default = 512
            The input embedding dimension.

        ffn_dim : int, default = 2048
            The dimension of the feedforward network.

        num_layers : int, default = 6
            The number of layers.

        num_heads : int, default = 8
            The number of attention heads.

        vocab_size : int, default = 33
            The size of the vocabulary.

        max_length : int, default = 320
            The maximum sequence length.

        dropout : float, default = 0.0
            The dropout probability.

        attention_dropout : float, default = 0.1
            The dropout probability for the attention weights.

        token_embedding_dropout : float, default = 0.0
            The dropout probability for the token embeddings.

        layer_norm_eps : float, default = 1e-5
            The epsilon value for the layer normalization.

        """
        super().__init__()
        self.config = config
        self.balm = BalmModel(config=self.config)
        self.lm_head = BalmLMHead(self.config.embed_dim, self.config.vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        # self.balm = BalmModel(
        #     embed_dim=embed_dim,
        #     ffn_dim=ffn_dim,
        #     num_heads=num_heads,
        #     num_layers=num_layers,
        #     vocab_size=vocab_size,
        #     max_length=max_length,
        #     dropout=dropout,
        #     attention_dropout=attention_dropout,
        #     token_embedding_dropout=token_embedding_dropout,
        #     layer_norm_eps=layer_norm_eps,
        #     padding_idx=padding_idx,
        # )
        # self.lm_head = BalmLMHead(embed_dim, vocab_size)
        # self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # @property
    # def num_parameters(self):
    #     return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> MaskedLMOutput:
        """
        Parameters
        ----------

        x : torch.Tensor
            The input tensor. Expected shape is (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            The output tensor. The shape is (batch_size, seq_len, vocab_size).
        """
        x = self.balm(
            input_ids,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=output_attentions,
        )
        if output_attentions:
            x, attn = x
        logits = self.lm_head(x)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        output = MaskedLMOutput(
            logits=logits,
            loss=masked_lm_loss,
        )
        if output_attentions:
            output.attentions = attn
        if output_hidden_states:
            output.hidden_states = x
        if return_dict:
            return output.as_dict()
        return output.as_tuple()
