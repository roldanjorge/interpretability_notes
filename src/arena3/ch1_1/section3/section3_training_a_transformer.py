# %%
# imports
import datasets
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate

from src.arena3.ch1_1.section2.config import Config
from src.arena3.ch1_1.section2.demo_transformer import DemoTransformer
from src.arena3.ch1_1.section3.training_args import TransformerTrainingArgs
from src.arena3.ch1_1.section3.transformer_trainer import TransformerTrainer
from src.arena3.utils.device import device

# %%
# Setup reference model
reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
)


# %%
# Create model
model_cfg = Config(
    debug=False,
    d_model=32,
    n_heads=16,
    d_head=2,
    d_mlp=32 * 4,
    n_layers=4,
    n_ctx=128,
    d_vocab=reference_gpt2.cfg.d_vocab,
)
model = DemoTransformer(model_cfg)

# %%
# Setup training arguments
args = TransformerTrainingArgs()

# %%
# Create data
dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
print(dataset)
print(dataset[0]["text"])


# %%
# tokenize and concatenate the dataset
tokenized_dataset = tokenize_and_concatenate(
    dataset,
    reference_gpt2.tokenizer,
    streaming=False,
    max_length=model.cfg.n_ctx,
    column_name="text",
    add_bos_token=True,
    num_proc=4,
)
dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
train_loader = DataLoader(
    dataset_dict["train"],
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset_dict["test"],
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# %%
first_batch = train_loader.dataset[: args.batch_size]

print(first_batch.keys())
print(first_batch["tokens"].shape)


# %%
# Training loop
# ===============================
# See the full run here: https://api.wandb.ai/links/dquarel/nrxuwnv7
model = DemoTransformer(model_cfg).to(device)
args = TransformerTrainingArgs()
trainer = TransformerTrainer(args, model, reference_gpt2.tokenizer, dataset_dict)
trainer.train()

# %%
