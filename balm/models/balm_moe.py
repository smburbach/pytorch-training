#!/usr/bin/python
# filename: balm_moe.py

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

from typing import Optional

# import lightning as L
import torch
import torch.nn as nn

from ..config import BalmMoEConfig
from ..embedding import RelativePositionalEmbedding
from ..loss import router_load_balancing_loss, router_z_loss
from ..modules import MaskedLMHead, MoETransformerLayer


class BalmMoE(nn.Module):
    """
    BALM Mixture of Experts model.
    """

    def __init__(self, config: BalmMoEConfig):
        super().__init__()
        self.config = config
        self.position_embedding_type = config.position_embedding_type
        # token embedding
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
        )
        # position embedding
        if config.position_embedding_type in ["relative_key", "relative_key_query"]:
            self.position_embeddings = RelativePositionalEmbedding(config.hidden_size)
        else:  # absolute
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
        # layers
        self.layers = nn.ModuleList(
            [MoETransformerLayer(config) for _ in range(config.num_layers)]
        )
        self.embedding_dropout = nn.Dropout(config.embedding_dropout_rate)
        self.final_norm = nn.LayerNorm(config.embed_dim)
        # LM head
        self.lm_head = MaskedLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.vocab_size,
            weight=self.embed_tokens.weight,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        return_dict: bool = True,
    ):
        """
        Parameters:
        -----------

        input_ids: torch.LomgTensor
            Tokenized input IDs

        attention_mask: torch.BoolTensor
            Attention mask

        output_attentions: bool
            Whether to output attention weights

        output_hidden_states: bool
            Whether to output hidden states

        output_router_logits: bool
            Whether to output router logits

        return_dict: bool
            Whether to return a dictionary of outputs (returns a tuple by default)


        Returns:
        --------
        output (tuple or dict):
            If `return_dict` is ``True``, the output is a ``dict`` of outputs:
                - last_hidden_state (torch.FloatTensor): last hidden state
                - router_z_loss (torch.FloatTensor): router z loss
                - router_aux_loss (torch.FloatTensor): router auxiliary loss
                - attentions (torch.FloatTensor): attention weights
                - hidden_states (torch.FloatTensor): hidden states
                - router_logits (torch.FloatTensor): router logits
            If `return_dict` is ``False``, the output is a ``tuple`` with the f0llowing elements:
                - last_hidden_state (torch.FloatTensor): last hidden state
                - attentions (torch.FloatTensor): attention weights
                - hidden_states (torch.FloatTensor): hidden states
                - router_logits (torch.FloatTensor): router logits
        """
        # init
        attn_weights = []
        hidden_states = {}
        router_logits = []
        expert_indexes = []

        # embeddings
        x = self.embed_scale * self.embed_tokens(input_ids)
        if self.position_embeddings_type in ["relative_key", "relative_key_query"]:
            x = self.position_embeddings(x)
        else:  # absolute embeddings
            position_ids = torch.arange(
                input_ids.size(1), dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeds = self.position_embeddings(position_ids)
            x = x + position_embeds
        x = self.embedding_dropout(x)

        # encoder
        x = x.transpose(0, 1)
        for layer_idx, layer in enumerate(self.layers, 1):
            x = layer(x, attention_mask, need_weights=output_attentions)
            if output_attentions:
                x, attn, router_tuple = x
                attn_weights.append(attn)
            else:
                x, router_tuple = x
            router_logits.append(router_tuple[0])
            expert_indexes.append(router_tuple[1])
            if output_hidden_states:
                hidden_states[layer_idx] = x.transpose(0, 1)
        x = self.final_norm(x)
        x = x.transpose(0, 1)

        # Compute the router losses (z_loss + auxiliary loss)
        cat_router_logits = torch.cat(router_logits, dim=1)
        cat_expert_indexes = torch.cat(expert_indexes, dim=1)
        router_probs = nn.Softmax(dim=-1)(cat_router_logits)
        z_loss = router_z_loss(cat_router_logits)
        aux_loss = router_load_balancing_loss(router_probs, cat_expert_indexes)

        # results
        outputs = [x]
        result = {
            "last_hidden_state": x,
            "router_z_loss": z_loss,
            "router_aux_loss": aux_loss,
        }
        if output_attentions:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            outputs.append(attentions)
        if output_hidden_states:
            result["hidden_states"] = hidden_states
            outputs.append(hidden_states)
        if output_router_logits:
            result["router_logits"] = cat_router_logits
            outputs.append(cat_router_logits)
        if return_dict:
            return result
        return tuple(outputs)


class BalmMoEForMaskedLM(nn.Module):
    """
    BALM Mixture of Experts model for Masked Language Modeling.
    """

    def __init__(self, config: BalmMoEConfig):
        super().__init__()
        self.config = config
        self.balm = BalmMoE(config)
        self.lm_head = MaskedLMHead(
            embed_dim=self.balm.embed_dim,
            output_dim=self.balm.vocab_size,
            weight=self.balm.embed_tokens.weight,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = True,
        return_dict: bool = True,
    ):
        """
        Args:
            input_ids (torch.LongTensor): tokenized input IDs
            attention_mask (torch.BoolTensor): attention mask
            return_dict (bool): return a dictionary of outputs
        """
        # encoder
        encoder_outputs = self.balm_moe(
            input_ids,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=True,
            return_dict=True,
        )
        x = encoder_outputs["last_hidden_state"]
        router_z_loss = encoder_outputs["router_z_loss"]
        router_aux_loss = encoder_outputs["router_aux_loss"]

        # LM head
        lm_logits = self.lm_head(x)

        # loss
        if labels is not None:
            loss_func = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_func(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            if output_router_logits:
                z_loss = self.router_z_loss_coef * (router_z_loss)
                aux_loss = self.router_aux_loss_coef * (router_aux_loss)
                loss = loss + z_loss + aux_loss
            encoder_outputs["loss"] = loss

        if return_dict:
            encoder_outputs["logits"] = lm_logits
            return encoder_outputs
        else:
            outputs = [
                lm_logits,
            ]
            if output_attentions:
                outputs.append(encoder_outputs["attentions"])
            if output_hidden_states:
                outputs.append(encoder_outputs["hidden_states"])
            if output_router_logits:
                outputs.append(encoder_outputs["router_logits"])
            return outputs
