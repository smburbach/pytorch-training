from balm.config import BalmConfig, BalmMoEConfig#, BalmExpertChoiceMoEConfig
from balm.data import load_dataset, DataCollator
from balm.models import (
    BalmForMaskedLM,
    BalmModel,
    BalmMoEForMaskedLM,
    #BalmExpertChoiceMoEForMaskedLM,
    #BalmHybridMoEForMaskedLM,
    #BalmMoERoPEForMaskedLM,
)
from balm.config import BalmConfig
from balm.tokenizer import Tokenizer
from balm.train import Trainer
import argparse
import pathlib
import deepspeed
import torch
from torch.utils.data import DataLoader
import math
from tqdm.auto import tqdm
from typing import Dict
from balm import DataCollator

def parser():
    parser = argparse.ArgumentParser()
    
    # add deepspeed parsing - following recommended method
    # https://deepspeed.readthedocs.io/en/latest/initialize.html#argument-parsing
    parser.add_argument(
        '--local_rank', 
        default=-1,
        type=int,
        help='local rank passed from distributed launcher'
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def place_inputs(collated: Dict):
    placed = {}
    for key, value in collated.items():
        value = value.to(torch.device("cuda"))
        placed[key] = value
    return placed

def main():
    # parse cl args
    args = parser()
    
    tokenizer = Tokenizer(vocab="./vocab.json")

    def remove_sep(txt):
        return txt.replace("</s>", "<cls><cls>")
    
    
    data_files = {
        "train": "./balm/test_data/test_1k.txt",
        "test": "./balm/test_data/test_1k.txt",
        "eval": "./balm/test_data/test_1k.txt",
    }
    
    dataset = load_dataset("text", data_files=data_files, preprocess_fn=remove_sep)

    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            padding=True,
            truncation=True,
            max_length=320,
        ),
        remove_columns="text"
    )

    collator = DataCollator(tokenizer=tokenizer)

    # # matched to ESM-2 8M
    config = BalmConfig(
        embed_dim=320,
        ffn_dim=320 * 4,
        num_layers=6,
        num_heads=20,
        vocab_size=tokenizer.vocab_size,
    )
    model = BalmForMaskedLM(config=config)

    # wrap model for deepspeed
    model_params = [p for p in model.parameters() if p.requires_grad]
    model_net, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model_params,
        config=args.deepspeed_config,
    )
    print(model_net)

    # dataloader
    total_batch_size = 32 * torch.cuda.device_count()
    train_dataloader = DataLoader(
        tokenized_dataset['train'],
        batch_size=total_batch_size,
        shuffle=True
    )
    # data collator
    data_collator = DataCollator(tokenizer)
    
    completed_steps = 0
    num_train_steps = 25
    num_epochs = math.ceil(num_train_steps * total_batch_size // len(tokenized_dataset['train']))
    pbar = tqdm(total=num_train_steps, unit="step", desc="Training")
    for epoch in range(num_epochs):
        for batch in train_dataloader:

            collated = data_collator(batch)
            inputs = place_inputs(collated)
            
            # forward pass
            outputs = model_net(
                input_ids=inputs["input_ids"],
                labels=inputs.get("labels", None),
                attention_mask=inputs.get("attention_mask", None),
                key_padding_mask=inputs.get("key_padding_mask", None),
            )

            loss = outputs["loss"]
            model_net.backward(loss)
            model_net.step()

            # logging
            completed_steps += 1
            pbar.update(1)
            if completed_steps % 5 == 0:
                print(completed_steps)

    # trainer = Trainer(
    #     model=model,
    #     data_collator=collator,
    #     train_dataset=tokenized_dataset["train"],
    #     eval_dataset=tokenized_dataset["eval"],
    #     output_dir="./training_runs/save_tests",
    #     logging_steps=5,
    #     eval_steps=100,
    #     warmup_steps=10,
    #     max_steps=25,
    #     per_device_train_batch_size=32,
    #     run_name="save_test_001",
    #     deepspeed=True,
    #     deepspeed_args=args,
    #     deepspeed_config=args.deepspeed_config
    # )
    # trainer.train()

if __name__ == "__main__":
    main()