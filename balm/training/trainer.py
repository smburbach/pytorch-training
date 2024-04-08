#!/usr/bin/python
# filename: trainer.py

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


import math
from functools import cached_property
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data import DataCollator, Dataset
from ..tokenizer import TokenizerBase
from .training_arguments import TrainingArguments
from .training_utils import set_seed


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        # args: TrainingArguments,
        data_collator: DataCollator,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        per_device_train_batch_size: Optional[int] = 1,
        per_device_eval_batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        logging_steps: Optional[int] = 100,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = 1,
        warmup_steps: Optional[int] = 0,
        weight_decay: Optional[float] = 0.01,
        learning_rate: Optional[float] = 4e-4,
        deepspeed: bool = False,
        deepspeed_config: Optional[str] = None,
        seed: Optional[int] = 42,
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        # callbacks: Optional[List[TrainerCallback]] = None,
    ):
        # if args is None:
        #     output_dir = "tmp_trainer"
        #     args = TrainingArguments(output_dir=output_dir)
        # self.args = args

        # seed gets set before anything else happens
        if seed is not None:
            set_seed(seed)
        self.model = model
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.epochs = epochs
        self.max_steps = max_steps
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.deepspeed = deepspeed
        self.deepspeed_config = deepspeed_config
        # if self.deepspeed:
        #     self.setup_deepspeed()

    @cached_property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @cached_property
    def device_count(self):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 1

    @property
    def n_epochs(self):
        if self.max_steps is None:
            return self.epochs
        return math.ceil(self.max_steps / len(self.train_dataset))

    @property
    def n_train_steps(self):
        if self.max_steps is not None:
            return self.max_steps
        else:
            return self.n_epochs * (
                len(self.train_dataset) // self.total_train_batch_size
            )

    @property
    def total_train_batch_size(self):
        return self.per_device_train_batch_size * self.device_count

    @property
    def total_eval_batch_size(self):
        if self.eval_dataset is None:
            return 0
        return self.per_device_eval_batch_size * self.device_count

    @property
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

    @property
    def eval_dataloader(self):
        if self.eval_dataset is None:
            return None
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

    def train(self):
        model, optimizer = self.wrap_model()
        completed_steps = 0
        pbar = tqdm(total=self.n_train_steps)
        for epoch in range(self.n_epochs):
            for i, batch in enumerate(self.train_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", None)
                if labels is not None:
                    labels = labels.to(self.device)
                attn_mask = batch.get("attention_mask", None)
                if attn_mask is not None:
                    attn_mask = attn_mask.to(self.device)
                key_padding_mask = batch.get("key_padding_mask", None)
                if key_padding_mask is not None:
                    key_padding_mask = key_padding_mask.to(self.device)
                optimizer.zero_grad()
                loss = model(
                    input_ids,
                    labels=labels,
                    attention_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )
                loss.backward()
                optimizer.step()
                completed_steps += 1
                pbar.update(1)
                if completed_steps % self.logging_steps == 0:
                    print(
                        f"step {completed_steps} | loss: {loss.item()} | lr: {optimizer.param_groups[0]['lr']}"
                    )
                if completed_steps % self.eval_steps == 0:
                    print("Evaluating")
                    #  TODO: evaluate
                if completed_steps % self.save_steps == 0:
                    print("Saving")
                    #  TODO: save
                if completed_steps >= self.n_train_steps:
                    print("Training complete")
                    break
        pbar.close()

    def wrap_model(self):
        if self.deepspeed:
            import deepspeed

            model_parameters = filter(
                lambda p: p.requires_grad, self.model.parameters()
            )
            model, optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=model_parameters,
                config="path_to_deepspeed_config.json",
            )
        else:
            model = self.model.to(self.device)
            if self.device_count > 1 and torch.cuda.is_available():
                model = nn.DataParallel(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        return model, optimizer

    # def setup_deepspeed(self):
    #     import deepspeed
