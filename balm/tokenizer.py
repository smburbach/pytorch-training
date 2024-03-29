import itertools
import json
import os
from typing import Dict, Iterable, Optional, Union

PROTEIN_TOKS = {
    "roberta": list("ACDEFGHIKLMNPQRSTVWY"),
    "esm": list("LAGVSERTIDPKQNFYMHWCXBUZO.-"),
}


class TokenizerMixin:
    """
    Mixin class for tokenizers. Provides methods for tokenization and encoding.
    """

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text) -> Iterable[str]:
        """
        Converts a string into a sequence of tokens by iteratively splitting on the tokenizer's list of tokens.

        Inspired by the ``tokenize`` method in the `HuggingFace Tokenizer`_ and `ESM Alphabet`_ classes.

        Parameters
        ----------
        text : str
            The sequence to be encoded.

        Returns
        -------
        Iterable[str]
            The list of tokens.

        .. _HuggingFace Tokenizer:
            https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py

        .. _ESM's Alphabet:
            https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/data.py#L91
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


class Tokenizer(TokenizerMixin):
    """
    Tokenizer class, heavily inspired by HuggingFace's Tokenizer_ class.
    Provides methods for loading a vocab, sequence tokenization, and encoding.

    Parameters
    ----------
    vocab : str, dict
        The vocabulary to be used. Can be either a JSON-formatted file,
        a directory containing a vocab.json file, or a ``dict``.

    prepend_bos : bool, optional
        Whether to prepend the beginning of sequence token. Default is True.

    append_eos : bool, optional
        Whether to append the end of sequence token. Default is False.

    cls_token : str, optional
        The beginning of sequence token. Default is "<cls>".

    pad_token : str, optional
        The padding token. Default is "<pad>".

    eos_token : str, optional
        The end of sequence token. Default is "<eos>".

    unk_token : str, optional
        The unknown token. Default is "<unk>".

    mask_token : str, optional
        The mask token. Default is "<mask>".

    .. _Tokenizer:
        https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
    """

    def __init__(
        self,
        vocab: Union[str, Dict[str, int]],
        prepend_bos: bool = True,
        append_eos: bool = True,
        cls_token: str = "<cls>",
        pad_token: str = "<pad>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        mask_token: str = "<mask>",
    ):
        self.vocab = self._process_vocab(vocab)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.vocab.keys())

        self.tok_to_idx = self.vocab

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

    def _process_vocab(self, vocab: Union[str, Dict[str, int]]) -> Dict[str, int]:
        if isinstance(vocab, str):
            if os.path.isfile(vocab):
                return json.load(open(vocab, "r"))
            elif os.path.isdir(vocab):
                return json.load(open(os.path.join(vocab, "vocab.json"), "r"))
            else:
                raise ValueError(
                    f"If vocab is a string, it must be a file or directory. {vocab} does not exist"
                )
        elif isinstance(vocab, dict):
            return vocab
        else:
            raise ValueError("Vocab must be a string or dictionary")

    @classmethod
    def from_pretrained(
        cls,
        vocab: Union[str, Dict[str, int]],
        prepend_bos: bool = True,
        append_eos: bool = True,
        cls_token: str = "<cls>",
        pad_token: str = "<pad>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        mask_token: str = "<mask>",
    ) -> "Tokenizer":
        """
        Loads a pretrained tokenizer from a vocab.

        Parameters
        ----------
        vocab : str, dict
            The vocabulary to be used. Can be either the name of a built-in
            vocab (e.g., "esm" or "roberta"), a JSON-formatted vocab file,
            a directory containing a vocab.json file, or a ``dict``.

        prepend_bos : bool, optional
            Whether to prepend the beginning of sequence token. Default is True.

        append_eos : bool, optional
            Whether to append the end of sequence token. Default is False.

        cls_token : str, optional
            The beginning of sequence token. Default is "<cls>".

        pad_token : str, optional
            The padding token. Default is "<pad>".

        eos_token : str, optional
            The end of sequence token. Default is "<eos>".

        unk_token : str, optional
            The unknown token. Default is "<unk>".

        mask_token : str, optional
            The mask token. Default is "<mask>".
        """
        if isinstance(vocab, str):
            if vocab.lower() == "esm":
                toks = ["<cls>", "<pad>", "<eos>", "<unk>"]
                toks += list(PROTEIN_TOKS["esm"])
                toks += ["<mask>"]
                return cls(
                    vocab={t: i for i, t in enumerate(toks)},
                    prepend_bos=True,
                    append_bos=True,
                    cls_token="<cls>",
                    pad_token="<pad>",
                    eos_token="<eos>",
                    unk_token="<unk>",
                    mask_token="<mask>",
                )
            elif vocab.lower() == "roberta":
                toks = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
                toks += list(PROTEIN_TOKS["roberta"])
                return cls(
                    vocab={t: i for i, t in enumerate(toks)},
                    prepend_bos=True,
                    append_bos=True,
                    cls_token="<s>",
                    pad_token="<pad>",
                    eos_token="</s>",
                    unk_token="<unk>",
                    mask_token="<mask>",
                )
            elif os.path.isdir(vocab):
                return cls(
                    vocab=json.load(open(os.path.join(vocab, "vocab.json"), "r")),
                    prepend_bos=prepend_bos,
                    append_eos=append_eos,
                    cls_token=cls_token,
                    pad_token=pad_token,
                    eos_token=eos_token,
                    unk_token=unk_token,
                    mask_token=mask_token,
                )

            elif os.path.isfile(vocab):
                return cls(
                    vocab=json.load(open(vocab, "r")),
                    prepend_bos=prepend_bos,
                    append_eos=append_eos,
                    cls_token=cls_token,
                    pad_token=pad_token,
                    eos_token=eos_token,
                    unk_token=unk_token,
                    mask_token=mask_token,
                )
            else:
                raise ValueError(f"Unknown vocab: {vocab}")
        return cls(
            vocab=vocab,
            prepend_bos=prepend_bos,
            append_eos=append_eos,
            cls_token=cls_token,
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            mask_token=mask_token,
        )


class Alphabet(TokenizerMixin):
    """
    Tokenizer class, heavily inspired by ESM's Alphabet_ class.
    Provides methods for loading pre-defined alphabets, sequence tokenization, and encoding.

    Parameters
    ----------
    standard_toks : Sequence[str], optional
        Standard tokens to be used. Default is PROTEIN_TOKS["esm"].

    prepend_toks : Sequence[str], optional
        Tokens to be prepended. Default is ("<cls>", "<pad>", "<eos>", "<unk>").

    append_toks : Sequence[str], optional
        Tokens to be appended. Default is ("<mask>",).

    prepend_bos : bool, optional
        Whether to prepend the beginning of sequence token. Default is True.

    append_eos : bool, optional
        Whether to append the end of sequence token. Default is False.

    cls_token : str, optional
        The beginning of sequence token. Default is "<cls>".

    pad_token : str, optional
        The padding token. Default is "<pad>".

    eos_token : str, optional
        The end of sequence token. Default is "<eos>".

    unk_token : str, optional
        The unknown token. Default is "<unk>".

    mask_token : str, optional
        The mask token. Default is "<mask>".

    .. _Alphabet:
        https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/data.py#L91
    """

    def __init__(
        self,
        standard_toks: Iterable[str] = PROTEIN_TOKS["esm"],
        prepend_toks: Iterable[str] = ("<cls>", "<pad>", "<eos>", "<unk>"),
        append_toks: Iterable[str] = ("<mask>",),
        prepend_bos: bool = True,
        append_eos: bool = False,
        cls_token: str = "<cls>",
        pad_token: str = "<pad>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        mask_token: str = "<mask>",
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
            prepend_toks: Iterable[str] = ("<s>", "</s>", "<pad>", "<unk>", "<mask>")
            append_toks: Iterable[str] = ()
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
