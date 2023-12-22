from _common import *

log = logging.getLogger(__name__)

from collections import defaultdict

import clip_dict
from torch.func import functional_call

from src.tasks.arithmetic import *
from src.utils import timeit_context


class BySampleProgram(clip_dict.DictLearnTTAProgram):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.result_dir = RESULTS_DIR / cfg.model / "layer_wise_dict"
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
        This function computes task vectors by merging the task vector with task-wise dictionary codings.

        Args:
            dict_codings (Tensor): A tensor containing dictionary codings for tasks.
            sample_idx (int): The index of the sample for which the task vectors are to be computed.

        Returns:
            dict: A dictionary where each key is a parameter key and the value is the computed task vector for that parameter.
        """
        model_task_vectors = {}
        coding = dict_codings[sample_idx].view(self.num_tasks, self.num_layers)
        for layer_idx, param_key in enumerate(self.task_vectors_as_dict[0].keys()):
            model_task_vectors[param_key] = 0
            for task_idx in range(self.num_tasks):
                model_task_vectors[param_key] += (
                    coding[task_idx, layer_idx]
                    * self.task_vectors_as_dict[task_idx][param_key]
                )
        return model_task_vectors

    def load_models(self, *, free_finetuned_models: bool = True):
        cfg = self.cfg
        self.num_tasks = len(cfg.seen_datasets)

        self.load_clip_models()
        with timeit_context("Computing task vectors"):
            self.task_vectors_as_dict = [
                state_dict_sub(
                    ft.state_dict(),
                    self.pretrained_model.state_dict(),
                    strict=False,
                    device=cfg.task_vector_device,
                )
                for i, ft in enumerate(self.finetuned_models)
            ]

        self.num_layers = len(self.task_vectors_as_dict[0])

        if free_finetuned_models:  # free finetuned models to save memory
            del self.finetuned_models

        self.load_dict_models(
            coding_size=self.num_tasks * self.num_layers
        )  # NOTE: coding_size = self.num_tasks * self.num_layers if layer-wise codings
        self.setup_preprocess()

        # setup forward model, this is used to perform inference
        self.forward_model = deepcopy(self.pretrained_model)
        for p in self.forward_model.parameters():
            p.requires_grad = False
        self.forward_model.eval()
        self.pretrained_sd = self.pretrained_model.state_dict()


class ByBatchProgram(BySampleProgram):
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
        dict_input = self.dict_preprocess(
            x, return_tensors="pt", do_rescale=False
        ).pixel_values.to(self.cfg.dict_mapping_device, non_blocking=True)
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
        (program := ByBatchProgram(cfg)).run()
    elif not cfg.eval_dict_tta and cfg.eval_dict:
        (program := BySampleProgram(cfg)).run()
    else:
        raise ValueError("either eval_dict or eval_dict_tta must be True")


if __name__ == "__main__":
    main()
