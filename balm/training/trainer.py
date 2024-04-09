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
import random
from functools import cached_property
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data import DataCollator, Dataset
from ..modules import MaskedLMOutput
from ..tokenizer import TokenizerBase
from .training_arguments import TrainingArguments
from .training_utils import get_scheduler


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
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        deepspeed: bool = False,
        deepspeed_config: Optional[str] = None,
        use_cpu: bool = False,
        seed: Optional[int] = 42,
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        # callbacks: Optional[List[TrainerCallback]] = None,
    ):
        # if args is None:
        #     output_dir = "tmp_trainer"
        #     args = TrainingArguments(output_dir=output_dir)
        # self.args = args

        # seed gets set before anything else happens
        self.seed = seed
        if seed is not None:
            self.set_seed(seed)
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
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.deepspeed = deepspeed
        self.deepspeed_config = deepspeed_config
        self.use_cpu = use_cpu
        # if self.deepspeed:
        #     self.setup_deepspeed()

    @cached_property
    def device(self):
        if torch.cuda.is_available() and not self.use_cpu:
            return torch.device("cuda")
        # elif torch.backends.mps.is_available() and not self.use_cpu:
        #     return torch.device("mps")
        else:
            return torch.device("cpu")

    @cached_property
    def device_count(self):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 1

    @property
    def num_epochs(self):
        if self.max_steps is None:
            # print("max_steps is None")
            return self.epochs
        # print(f"max_steps is {self.max_steps}")
        # print(f"train_dataset length is {len(self.train_dataset)}")
        # print(f"num_epochs is {math.ceil(self.max_steps / len(self.train_dataset))}")
        return math.ceil(
            self.max_steps * self.total_train_batch_size / len(self.train_dataset)
        )

    @property
    def num_train_steps(self):
        if self.max_steps is not None:
            return self.max_steps
        else:
            return (
                self.num_epochs * len(self.train_dataset) // self.total_train_batch_size
            )

    @property
    def num_warmup_steps(self):
        if self.warmup_steps < 1:  # warmup ratio
            return int(self.num_train_steps * self.warmup_steps)
        return self.warmup_steps

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
            batch_size=self.total_train_batch_size,
            shuffle=True,
            # collate_fn=self.data_collator,
        )

    @property
    def eval_dataloader(self):
        if self.eval_dataset is None:
            return []
        return DataLoader(
            self.eval_dataset,
            batch_size=self.total_eval_batch_size,
            shuffle=False,
            # collate_fn=self.data_collator,
        )

    def train(self):
        model, optimizer = self.wrap_model()
        model.train()

        scheduler = get_scheduler(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_train_steps,
        )

        completed_steps = 0
        pbar = tqdm(total=self.num_train_steps)
        for epoch in range(self.num_epochs):
            for batch in self.train_dataloader:
                optimizer.zero_grad()
                collated = self.data_collator(batch)
                inputs = self.place_inputs(collated)
                outputs = model(
                    input_ids=inputs["input_ids"],
                    labels=inputs.get("labels", None),
                    attention_mask=inputs.get("attention_mask", None),
                    key_padding_mask=inputs.get("key_padding_mask", None),
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                completed_steps += 1
                pbar.update(1)

                # logging
                if completed_steps % self.logging_steps == 0:
                    self.print_loss(
                        steps=completed_steps,
                        outputs=outputs,
                        lr=optimizer.param_groups[0]["lr"],
                        num_train_steps=self.num_train_steps,
                    )

                # eval
                if (
                    self.eval_steps is not None
                    and self.eval_dataset is not None
                    and completed_steps % self.eval_steps == 0
                ):
                    print("Evaluating")
                    #  TODO: evaluate

                # save
                if (
                    self.save_steps is not None
                    and completed_steps % self.save_steps == 0
                ):
                    print("Saving")
                    #  TODO: save

                # done!
                if completed_steps >= self.num_train_steps:
                    print("Training complete")
                    break
        pbar.close()

    def evaluate(self, model: nn.Module):
        pass

    def wrap_model(self) -> Tuple[nn.Module, torch.optim.Optimizer]:
        if self.deepspeed:
            import deepspeed

            model_params = [p for p in self.model.parameters() if p.requires_grad]
            model, optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=model_params,
                config="path_to_deepspeed_config.json",
            )
        else:
            model = self.model.to(self.device)
            if self.device_count > 1 and torch.cuda.is_available():
                model = nn.DataParallel(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
            )
        return model, optimizer

    def place_inputs(self, collated: Dict):
        for key, value in collated.items():
            if value is not None:
                value = value.to(self.device)
        return collated

    @staticmethod
    def set_seed(seed: int):
        """
        Helper function for reproducible behavior to set the seed in
        ``random``, ``numpy``, and ``torch`` (if installed).

        Args:
            seed (`int`): The seed to set.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # safe even if cuda isn't available

    @staticmethod
    def print_loss(
        steps: int,
        outputs: MaskedLMOutput,
        lr: float,
        num_train_steps: Optional[int] = None,
    ):
        if num_train_steps is not None:
            total_spaces = len(str(num_train_steps))
            spaces = " " * (total_spaces - len(str(steps)))
        else:
            spaces = ""
        log_str = f"step {steps}{spaces} | loss: {outputs.loss.item():0.4f}"
        if outputs.lm_loss is not None:
            log_str += f" | lm_loss: {outputs.lm_loss.item():0.4f}"
        if outputs.router_z_loss is not None:
            log_str += f" | router_z_loss: {outputs.router_z_loss.item():0.4f}"
        if outputs.router_aux_loss is not None:
            log_str += f" | router_aux_loss: {outputs.router_aux_loss.item():0.4f}"
        log_str += f" | lr: {lr:0.6f}"
        print(log_str)
