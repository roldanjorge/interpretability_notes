from dataclasses import dataclass


@dataclass
class TransformerTrainingArgs:
    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    wandb_project: str | None = "day1-demotransformer-jr"
    wandb_name: str | None = "run_2026_02_28_v1"


# @dataclass
# class TransformerTrainingArgs:
#     batch_size: int = 32
#     epochs: int = 10
#     max_steps_per_epoch: int = 500
#     lr: float = 1e-3
#     weight_decay: float = 1e-2
#     wandb_project: str | None = "day1-demotransformer-jr"
#     wandb_name: str | None = "run_2026_02_22_v1"
