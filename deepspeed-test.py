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

    # matched to ESM-2 8M
    config = BalmConfig(
        embed_dim=320,
        ffn_dim=320 * 4,
        num_layers=6,
        num_heads=20,
        vocab_size=tokenizer.vocab_size,
    )
    model = BalmForMaskedLM(config=config)

    trainer = Trainer(
        model=model,
        data_collator=collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        output_dir="./training_runs/save_tests",
        # epochs=1,
        logging_steps=5,
        eval_steps=100,
        warmup_steps=10,
        max_steps=25,
        # save_steps=15,
        per_device_train_batch_size=32,
        # use_cpu=True,
        # use_wandb=True,
        # wandb_project="test_wandb_logging",
        # wandb_entity="bryanbriney",
        run_name="save_test_001",
        deepspeed=True,
        deepspeed_args=args,
        deepspeed_config=args.deepspeed_config
    )
    trainer.train()

if __name__ == "__main__":
    main()