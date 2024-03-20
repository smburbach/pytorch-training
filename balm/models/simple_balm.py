from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbeddings(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super(RotaryEmbeddings, self).__init__()
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


class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.sigmoid(x1) * x2


class RoformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        forward_expansion: int,
        max_len: int,
        dropout: float,
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
        super(RoformerBlock, self).__init__()
        self.rotary_embedding = RotaryEmbeddings(embed_dim, max_len)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=heads, dropout=dropout, batch_first=True
        )

        # SwiGLU
        hidden_dim = embed_dim * forward_expansion
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            SwiGLU(),
            nn.Linear(hidden_dim // 2, embed_dim),  # adjusted for SwiGLU
        )
        # # ReLU
        # self.feed_forward = nn.Sequential(
        #     nn.Linear(embed_dim, forward_expansion * embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(forward_expansion * embed_dim, embed_dim),
        # )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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
        attn_output, _ = self.attention(query_rot, key_rot, value_rot, attn_mask=mask)
        attn_output = query + self.dropout(attn_output)

        # pre-norm
        attn_output_norm = self.norm2(attn_output)

        # feedforward
        ff_output = self.feed_forward(attn_output_norm)
        ff_output = attn_output + self.dropout(ff_output)

        return ff_output


class Balm2Model(nn.Module):
    def __init__(
        self,
        embed_dim: int = 320,
        heads: int = 20,
        forward_expansion: int = 4,
        num_layers: int = 6,
        vocab_size: int = 25,
        max_len: int = 320,
        dropout: float = 0.1,
    ):
        """

        Parameters
        ----------
        embed_dim : int
            The input embedding dimension.

        heads : int
            The number of attention heads.

        forward_expansion : int
            The expansion factor for the feedforward network.

        num_layers : int
            The number of layers.

        vocab_size : int
            The size of the vocabulary.

        max_len : int
            The maximum sequence length.

        dropout : float
            The dropout probability.

        """
        super(Balm2Model, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [
                RoformerBlock(
                    embed_dim, heads, embed_dim, forward_expansion, max_len, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
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

        for layer in self.layers:
            x = layer(x, x, x)

        x = self.final_layer_norm(x)

        return x
