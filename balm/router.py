#!/usr/bin/python
# filename: router.py

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


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterBase(nn.Module):
    """
    Base class for routers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        expert_capacity: int,
        dtype: str = "float32",
        bias: bool = False,
        jitter: float = 0.0,
        num_routable_experts: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.dtype = getattr(torch, dtype)
        self.bias = bias
        self.jitter = jitter
        self.classifier = nn.Linear(
            self.embed_dim,
            num_routable_experts
            if num_routable_experts is not None
            else self.num_experts,
            bias=self.bias,
            dtype=self.dtype,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

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


class TopKRouter(RouterBase):
    """
    This router uses the "token choice of top-k experts" strategy. For example, if k=1, this
    replicates the top-1 routing strategy introduced in the `Switch Transformers`_ paper.
    Alternatively, if k=2, this replicates the top-2 routing strategy introduced in the `GShard`_
    paper. Tokens are routed to their expert of choice until the expert's `expert_capacity` is
    reached.

    .. note::
        There is no guarantee that each token will be processed by an expert,
        or that every expert will receive at least one token.

    If tokens are routed to an expert which is above capacity, they are not processed by any expert
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

    .. _GShard:
        https://arxiv.org/abs/2006.16668
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        expert_capacity: int,
        top_k: int = 1,
        dtype: str = "float32",
        bias: bool = False,
        jitter: float = 0.0,
        ignore_padding_tokens: bool = True,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_experts=num_experts,
            expert_capacity=expert_capacity,
            dtype=dtype,
            bias=bias,
            jitter=jitter,
        )
        self.top_k = top_k
        self.ignore_padding_tokens = ignore_padding_tokens

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-k experts.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        top_k : int
            Number of top experts to route each token to.

        Returns:
        --------
        expert_indices : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) indicating
            which experts the token should be routed to.

        router_probabilities : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) containing
            the router probabilities.

        router_logits : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) containing
            the router logits.
        """
        router_probs, router_logits = self._compute_router_probabilities(x)
        top_k_values, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        expert_indices = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(
            dim=-2
        )

        # mask tokens if their desired experts are above capacity
        token_priority = torch.cumsum(expert_indices, dim=-2)
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_indices = expert_indices * expert_capacity_mask

        # get the probabilities of the top-choice experts for each token
        router_probs = top_k_values * expert_indices

        return expert_indices, router_probs, router_logits


class ExpertChoiceRouter(RouterBase):
    """
    Router that selects top-k tokens for each expert and has shared experts that process all tokens.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        expert_capacity: int,
        num_shared_experts: int = 1,  # Number of shared experts
        dtype: str = "float32",
        bias: bool = False,
        jitter: float = 0.0,
        ignore_padding_tokens: bool = True,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_experts=num_experts,
            expert_capacity=expert_capacity,
            dtype=dtype,
            bias=bias,
            jitter=jitter,
            num_routable_experts=num_experts - num_shared_experts,
        )
        self.num_shared_experts = num_shared_experts
        self.ignore_padding_tokens = ignore_padding_tokens

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts, selecting top-k tokens for each expert, and route all tokens to shared experts.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
        --------
        expert_mask : torch.Tensor
            Binary mask tensor of shape (batch_size, sequence_length, num_experts) indicating
            which tokens are selected for each expert and which are processed by shared experts.

        router_probabilities : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) containing
            the router probabilities.

        router_logits : torch.Tensor
            Tensor of shape (batch_size, sequence_length, num_experts) containing
            the router logits.
        """
        router_probs, router_logits = self._compute_router_probabilities(x)
        expert_mask = torch.zeros_like(router_probs)

        # Select top-k tokens for each expert
        for i in range(self.num_experts - self.num_shared_experts):
            _, top_k_indices = torch.topk(
                router_probs[..., i], k=self.expert_capacity, dim=1
            )
            expert_mask.scatter_(1, top_k_indices.unsqueeze(-1), 1, reduce="add")

        # Ensure that the mask is binary
        expert_mask = expert_mask.clamp(max=1)

        # Add shared experts processing all tokens
        if self.num_shared_experts > 0:
            shared_expert_mask = torch.ones_like(
                router_probs[..., : self.num_shared_experts]
            )
            expert_mask = torch.cat((shared_expert_mask, expert_mask), dim=-1)

        return expert_mask, router_probs, router_logits
