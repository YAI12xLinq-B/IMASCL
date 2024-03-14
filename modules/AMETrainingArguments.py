from transformers import TrainingArguments
from transformers.utils import add_start_docstrings
from transformers.generation import GenerationConfig

from dataclasses import dataclass, field
from dataclasses import field

@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class AMETrainingArguments(TrainingArguments):
    cross_acc_sampling: bool = field(
        default=False,
        metadata={"help": "Whether or not to use cross accelerator sampling. use in-batch sampling in default "},
    )

    dataset_pseudo: str = field(
        default="None",
        metadata={"help": "Path of pseudo dataset, type None if you don't want to use AME"},
    )

    margin: float = field(
        default=0.3,
        metadata={"help": "Margin of positive pair in contrastive learning"},
    )

    positive_threshold: float = field(
        default=0.95,
        metadata={"help": "Threshold for pseudo-label to predict as positive pair in contrastive learning, use -1 not to use "},
    )

    loss_type: str = field(
        default = "CS",
        metadata={"help": "weight fucntion to forward pseudo-label"},
    )

    tau: float = field(
        default=0.1,
        metadata={"help": "Temperature to use at contrastive learning"},
    )

    w_func: str = field(
        default="raw",
        metadata={"help": "Fucntion to forward pseudo-label"},
    )

    gpu_num: int = field(
        default=1,
        metadata={"help": "The number of gpus that are used in training"},
    )

    embed_model_dir: str = field(
        default=None,
        metadata={"help": "Only activated when dataset_embed = None, model to be used with on-the-fly method, type self if you want to use the training model"}
    )

    embed_direction: str = field(
        default="left",
        metadata={"help": "embedding direction, left/right/both/input, if input, --langs input order will be the priority, place higher priority at first to use this"}
    )

    train_mono_space: str = field(
        default = False,
        metadata={"help": "also train monolingual space"}
    )

    small_batch_size: int = field(
        default=None,
        metadata={"help": "Divide similar matrix into (small_batch_size x small_batch_size) when computing loss"}
    )
    metrics_for_save_model: tuple = field(
        default=("eval_loss", "eval_tatoeba_acc_avg"),
        metadata={"help": "setting for multiple best_model criteria"}
    )
    metrics_larger_better: tuple = field(
        default=(False, True),
        metadata={"help": "setting for multiple best_model criteria, True means larger metrics in metrics for save model better"}
    )
    dataset_sentence_idx_dict: dict = field(
        default=None,
        metadata={"help": "index dictionary for train language pairs. it's value has (start, end, end after padding)"}
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values and `GenerationConfig` by dictionaries (for JSON
        serialization support). It obfuscates the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = super().to_dict()
        for k, v in d.items():
            if isinstance(v, GenerationConfig):
                d[k] = v.to_dict()
        return d