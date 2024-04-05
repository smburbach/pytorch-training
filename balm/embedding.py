#!/usr/bin/python
# filename: embedding.py

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


import torch
from torch import nn


class RelativePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=320):
        super(RelativePositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super(RotaryPositionalEmbeddings, self).__init__()
        self.dim = dim
        self.max_len = max_len
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    def get_positional_embeddings(self, x: torch.Tensor):
        """
        Generates the sinusoidal positional embeddings.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor. Expected shape is (batch_size, seq_len, dim).

        Returns
        -------
        torch.Tensor
            The positional embeddings. The shape is (seq_len, dim).
        """
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", positions, self.inv_freq)
        return torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

    def apply_rotary_embeddings(self, x: torch.Tensor):
        """
        Applies rotary embeddings to the input tensor x.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor. Expected shape is (batch_size, seq_len, dim).

        Returns
        -------
        torch.Tensor
            The input tensor with rotary embeddings applied. The shape is (batch_size, seq_len, dim).

        """
        pos_emb = self.get_positional_embeddings(x)
        s, c = pos_emb[:, : self.dim // 2], pos_emb[:, self.dim // 2 :]
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        x_rot = torch.cat(((x1 * c) + (x2 * s), (-x1 * s) + (x2 * c)), dim=-1)
        return x_rot

    def forward(self, x: torch.Tensor):
        return self.apply_rotary_embeddings(x)
