from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import EsmTokenizer


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


class BalmLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = nn.GELU(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias
        return x


class BalmModel(nn.Module):
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
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [
                RoformerBlock(embed_dim, heads, forward_expansion, max_len, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class BalmForMaskedLM(nn.Module):
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
        super().__init__()
        self.balm = BalmModel(
            embed_dim=embed_dim,
            heads=heads,
            forward_expansion=forward_expansion,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_len=max_len,
            dropout=dropout,
        )
        self.lm_head = BalmLMHead(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.balm(x)
        x = self.lm_head(x)
        return x


class BalmTokenizer(EsmTokenizer):
    """
    Tokenizer for Balm2 model. Follows the same API as the ESM tokenizer.

    Parameters
    ----------
    vocab_file : str
        The path to the JSON-formatted vocabulary file.

    unk_token : str, optional
        The unknown token. Default is "<unk>".

    cls_token : str, optional
        The classification token. Default is "<cls>".

    pad_token : str, optional
        The padding token. Default is "<pad>".

    mask_token : str, optional
        The mask token. Default is "<mask>".

    eos_token : str, optional
        The end-of-sequence token. Default is "<eos>".

    **kwargs
        Additional keyword arguments passed to the parent class.


    """

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<unk>",
        cls_token: str = "<cls>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        eos_token: str = "<eos>",
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            **kwargs,
        )


model = BalmForMaskedLM()

# Assuming you have your dataset ready and DataLoader created
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Assuming you have defined your loss function and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0 and batch_idx > 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {total_loss / log_interval:.6f}"
            )
            total_loss = 0.0


def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation set: Average loss: {avg_loss:.4f}")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10
log_interval = 100

for epoch in range(1, num_epochs + 1):
    train(model, train_loader, loss_func, optimizer, epoch)
    evaluate(model, val_loader, loss_func)
