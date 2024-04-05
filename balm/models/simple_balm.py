import itertools
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, List, Optional, Sequence, Tuple, Union

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


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        ff_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        max_len: int = 320,
        dropout: float = 0.1,
    ):
        """
        Simple Transformer
        """
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) + self.positional_encoding[:, : x.shape[1], :]
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return x


class SimpleTransformerForMaskedLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        ff_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        max_len: int = 320,
        dropout: float = 0.1,
    ):
        """
        Simple Transformer for masked language modeling.
        """
        super(SimpleTransformerForMaskedLM, self).__init__()
        self.encoder = SimpleTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout,
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.lm_head.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        logits = self.lm_head(x)
        return logits


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


PROTEIN_TOKS = {
    "roberta": list("ACDEFGHIKLMNPQRSTVWY"),
    "esm": list("LAGVSERTIDPKQNFYMHWCXBUZO.-"),
}


class Alphabet(object):
    """
    Modified version of ESM's Alphabet_ class. Provides methods for loading pre-defined alphabets,
    sequence tokenization, and encoding.


    .. _Alphabet:
        https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/data.py#L91
    """

    def __init__(
        self,
        standard_toks: Sequence[str] = PROTEIN_TOKS["esm"],
        prepend_toks: Sequence[str] = ("<cls>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<mask>",),
        prepend_bos: bool = True,
        append_eos: bool = False,
        cls_token: str = "<cls>",
        pad_token: str = "<pad>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        mask_token: str = "<mask>",
        use_msa: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        # for i in range((8 - (len(self.all_toks) % 8)) % 8):
        #     self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx[unk_token]
        self.padding_idx = self.get_idx(pad_token)
        self.cls_idx = self.get_idx(cls_token)
        self.mask_idx = self.get_idx(mask_token)
        self.eos_idx = self.get_idx(eos_token)
        self.all_special_tokens = [
            eos_token,
            unk_token,
            pad_token,
            cls_token,
            mask_token,
        ]
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def __call__(self, text: str):
        return self.encode(text)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        if name.lower() in ("balm", "roberta"):
            standard_toks = PROTEIN_TOKS["roberta"]
            prepend_toks: Tuple[str, ...] = ("<s>", "</s>", "<pad>", "<unk>", "<mask>")
            append_toks: Tuple[str, ...] = ()
            cls_token = "<s>"
            pad_token = "<pad>"
            eos_token = "</s>"
            unk_token = "<unk>"
            mask_token = "<mask>"
            prepend_bos = True
            append_eos = True
        elif name.lower() in ("esm"):
            standard_toks = PROTEIN_TOKS["esm"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            cls_token = "<cls>"
            pad_token = "<pad>"
            eos_token = "<eos>"
            unk_token = "<unk>"
            mask_token = "<mask>"
            prepend_bos = True
            append_eos = True
        else:
            raise ValueError("Unknown architecture selected")
        return cls(
            standard_toks,
            prepend_toks,
            append_toks,
            cls_token,
            pad_token,
            eos_token,
            unk_token,
            mask_token,
            prepend_bos,
            append_eos,
        )

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, spl_text in enumerate(split_text):
                # strip white space from left and right sides of the splits
                spl_text = spl_text.strip()
                # parse text split
                if i == 0 and not spl_text:
                    # if the first item in the split is an empty string,
                    # we should add the token used to split:
                    #
                    # "ABC".split("A") --> ["", "B", "C"]
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if spl_text:
                        result.append(spl_text)
                    else:
                        # edge case with a single token input that matches the splitting token:
                        #
                        # "A".split("A") --> ["", ""]
                        #
                        # this could result in duplication of he splitting token, since we've
                        # already replaced the first empty string:
                        pass
                else:
                    if spl_text:
                        result.append(spl_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text
            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(
        self,
        text: str,
        max_length: int = 320,
        pad_to_max_length: bool = True,
    ):
        """
        Encodes a string in a sequence of tokens, using the tokenizer.

        Parameters
        ----------
        text : str
            The sequence to be encoded.

        max_length : int, optional
            The maximum sequence length. Default is 320.

        pad_to_max_length : bool, optional
            Whether to pad the sequence to the maximum length. Default is True.

        threads : int, optional
            The number of threads to use. Default is 1.

        Returns
        -------
        List[int]
            The list of token indices.
        """
        encoded = [self.tok_to_idx[tok] for tok in self.tokenize(text)]

        # prepend bos token, if necessary
        if self.prepend_bos and encoded[0] != self.cls_idx:
            encoded = [self.cls_idx] + encoded

        # truncate and add eos token if necessary
        if len(encoded) >= max_length:
            encoded = encoded[:max_length]
            if self.append_eos:
                encoded = encoded[:-1] + [self.eos_idx]

        # pad and add eos token if necessary
        else:
            if self.append_eos:
                if pad_to_max_length:
                    encoded += [self.padding_idx] * (max_length - len(encoded) - 1)
                    encoded.append(self.eos_idx)
                else:
                    encoded = encoded.append(self.eos_idx)
            else:
                if pad_to_max_length:
                    encoded += [self.padding_idx] * (max_length - len(encoded))
        return encoded


with ProcessPoolExecutor as executor:
    encoded = executor.map(self.tok_to_idx.get, self.tokenize(text))
    encoded = list(encoded)


def mask_tokens(
    self,
    inputs: torch.Tensor,
    tokenizer: Any,
    mlm_probability: float = 0.15,
    special_tokens_mask: Optional[Union[torch.Tensor, Any]] = None,
) -> Tuple[Any, Any]:
    """
    Mask tokens for language modeling. By default, 15% of tokens are selected for masking, of which
    80% are masked, 10% are replaced with a random token, and 10% are unchanged.

    Parameters
    ----------
    inputs : torch.Tensor
        The input tensor. Expected shape is (batch_size, seq_len).

    tokenizer : Any
        The tokenizer.

    mlm_probability : float, optional
        The masking probability. Default is 0.15.

    special_tokens_mask : Optional[Union[torch.Tensor, Any]], optional
        The special tokens mask. Default is None.

    Returns
    -------
    Tuple[Any, Any]
        The masked input tensor and the labels tensor.
    """
    labels = inputs.clone()
    # randomly select tokens for masking
    probability_matrix = torch.full(labels.shape, mlm_probability)  # uniform
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # ignore non-masked tokens for loss calculation

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


model = BalmForMaskedLM()

# Assuming you have your dataset ready and DataLoader created
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Assuming you have defined your loss function and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
def train(model, train_loader, loss_func, optimizer, tokenizer, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        inputs, labels = mask_tokens(batch["input_ids"], tokenizer)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        # for batch_idx, (inputs, targets) in enumerate(train_loader):
        #     inputs, targets = inputs.to(device), targets.to(device)
        #     optimizer.zero_grad()
        #     outputs = model(inputs)
        #     loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        #     loss.backward()
        #     optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0 and batch_idx > 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {total_loss / log_interval:.6f}"
            )
            total_loss = 0.0


def evaluate(model, val_loader, tokenizer, loss_func):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = mask_tokens(batch["input_ids"], tokenizer)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

    # with torch.no_grad():
    #     for inputs, targets in val_loader:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model(inputs)
    #         loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    #         total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation set: Average loss: {avg_loss:.4f}")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10
log_interval = 100

for epoch in range(1, num_epochs + 1):
    train(model, train_loader, loss_func, optimizer, tokenizer, epoch)
    evaluate(model, val_loader, loss_func)
