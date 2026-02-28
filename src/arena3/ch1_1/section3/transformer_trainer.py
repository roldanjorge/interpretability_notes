from jaxtyping import Float, Int
import numpy as np
import torch as t
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
import wandb

from src.arena3.ch1_1.section2.demo_transformer import DemoTransformer
from src.arena3.ch1_1.section3.training_args import TransformerTrainingArgs
from src.arena3.ch1_1.section3.transformer_sampler import TransformerSampler
from src.arena3.utils.utils import get_log_probs


class TransformerTrainer:
    def __init__(
        self,
        args: TransformerTrainingArgs,
        model: DemoTransformer,
        tokenizer: AutoTokenizer,
        dataset_dict: dict[str, t.utils.data.Dataset],
    ):
        super().__init__()
        self.model = model
        self.args = args
        self.sampler = TransformerSampler(self.model, tokenizer)
        self.optimizer = t.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.step = 0

        self.train_loader = DataLoader(
            dataset_dict["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            dataset_dict["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        """
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        """
        loss = None

        logits = self.model(batch["tokens"])
        pred_log_probs = get_log_probs(logits, batch["tokens"])
        loss = -pred_log_probs.mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        wandb.log({"loss": loss.item(), "step": self.step})
        self.step += 1
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """
        Evaluate the model on the test set and return the accuracy.
        """
        self.model.eval()
        #
        # YOUR CODE HERE - fill in the `evaluate` method
        #
        accuracy = np.nan
        total_correct = 0
        for i, batch in enumerate(self.test_loader):
            logits = self.model(batch["tokens"])
            # pred_log_probs = get_log_probs(logits, batch["tokens"])
            pred_tokens = logits.argmax(dim=-1)
            correct = (pred_tokens == batch["tokens"][:, 1:]).sum().item()
            total_correct += correct

        accuracy = total_correct / len(self.test_loader.dataset)
        self.model.train()
        return accuracy

    def train(self):
        """
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        """
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        accuracy = np.nan

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs)

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(
                    f"Epoch {epoch + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                )
                if i >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()
            sample_text = self.sampler.sample("Once upon a time", max_tokens_generated=50)
            print(sample_text)

        wandb.finish()
