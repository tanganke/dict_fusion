from _common import *

log = logging.getLogger(__name__)

from collections import defaultdict

import clip_dict
import torchvision.transforms
from clip_checkpoint_path import (
    CHECKPOINT_DIR,
    finetuned_model_path,
    pretrained_model_path,
)
from torch.func import functional_call
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, ResNetForImageClassification

from src.adamerging import softmax_entropy
from src.clip_eval import eval_single_dataset, eval_single_dataset_preprocess_head
from src.datasets.common import maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from src.tasks import check_parameterNamesMatch
from src.tasks.arithmetic import *
from src.tasks.task_vector import StateDict, TaskVector
from src.ties_merging_utils import state_dict_to_vector, vector_to_state_dict
from src.utils import timeit_context


class Program(clip_dict.Program):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.result_dir = RESULTS_DIR / cfg.model / "task_wise_dict"
        if cfg.version is not None:
            self.result_dir = self.result_dir / f"version_{cfg.version}"
        self.result_dir.mkdir(exist_ok=True, parents=True)
        self.result_path = self.result_dir / "results.csv"

    def compute_task_vectors(
        self,
        dict_codings: Tensor,
        sample_idx: int,
    ):
        """
        Merges the task vector with task-wise dictionary coding.

        It first retrieves the task vectors as a dictionary and the dictionary codings for each task.
        It then computes the weighted sum of the task vectors, using the dictionary codings as weights.
        The weighted sum is computed on the device specified in the configuration for task vectors.

        Args:
            dict_codings (Tensor): The dictionary codings for each task.
            sample_idx (int): The index of the sample in the batch.

        Returns:
            Tensor: The merged task vector.
        """
        model_task_vector = state_dict_weighted_sum(
            self.task_vectors_as_dict,
            dict_codings[sample_idx],
        )
        return model_task_vector


class Program2(Program):
    def compute_task_vectors(
        self,
        dict_codings: Tensor,
        sample_idx: int,
    ):
        model_task_vector = state_dict_weighted_sum(
            self.task_vectors_as_dict,
            dict_codings[sample_idx],
        )
        return model_task_vector

    @functools.cache
    def forward_device(self, task_idx: int):
        cfg = self.cfg

        if not isinstance(cfg.forward_devices, (list, ListConfig)):
            return torch.device(cfg.forward_devices)
        else:
            num_forward_devices = len(cfg.forward_devices)
            assert (
                self.num_tasks % num_forward_devices == 0
            ), "batch size must be divisible by the number of forward devices"
            forward_device_idx = task_idx // (self.num_tasks // num_forward_devices)
            return torch.device(cfg.forward_devices[forward_device_idx])

    def model_forward(self, x, *, task_idx: int, task_name: str):
        cfg = self.cfg

        # compute the dictionary codings
        dict_input = self.dict_preprocess(x, return_tensors="pt").pixel_values.to(
            self.cfg.dict_mapping_device, non_blocking=True
        )
        dict_features = self.dict_feature_extractor(dict_input)
        dict_codings = self.dict_mapping(dict_features).to(cfg.task_vector_device)

        # compute the model logits
        model_input = self.model_preprocess(x)  # still on CPU
        model_features = []

        # * NOTE: to reduce the memory usage, we compute the model features batch by batch in Program2
        dict_codings = dict_codings.mean(dim=0, keepdim=True)
        model_task_vector = self.compute_task_vectors(dict_codings, 0)
        model_task_vector = {
            k: v.to(self.forward_device(task_idx)) for k, v in model_task_vector.items()
        }
        model_sd = state_dict_add(
            self.pretrained_sd_on_device(self.forward_device(task_idx)),
            model_task_vector,
        )
        model_features = functional_call(
            self.forward_model,
            model_sd,
            model_input.to(self.forward_device(task_idx)),
        ).to(self.forward_device(0), non_blocking=True)

        # for sample_idx in range(x.size(0)):
        #     model_task_vector = self.compute_task_vectors(dict_codings, sample_idx)
        #     model_task_vector = {
        #         k: v.to(self.forward_device(sample_idx))
        #         for k, v in model_task_vector.items()
        #     }
        #     model_sd = state_dict_add(
        #         self.pretrained_sd_on_device(self.forward_device(sample_idx)),
        #         model_task_vector,
        #     )
        #     _model_feature = functional_call(
        #         self.forward_model,
        #         model_sd,
        #         model_input[sample_idx : sample_idx + 1].to(
        #             self.forward_device(sample_idx)
        #         ),
        #     ).to(self.forward_device(0), non_blocking=True)
        #     model_features.append(_model_feature)
        # model_features = torch.cat(model_features)
        model_logits = self.classification_heads[task_name](model_features)
        return model_logits  # on the first forward device


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="clip_dict",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if cfg.eval_dict_tta and not cfg.eval_dict:
        (program := Program2(cfg)).run()
    elif not cfg.eval_dict_tta and cfg.eval_dict:
        (program := Program(cfg)).run()
    else:
        raise ValueError("either eval_dict or eval_dict_tta must be True")


if __name__ == "__main__":
    main()
