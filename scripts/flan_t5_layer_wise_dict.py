from _common import *

log = logging.getLogger(__name__)

from flan_t5_dict import DictLearnTTAProgram
from torch.func import functional_call

from src.tasks.arithmetic import *


class BySampleProgram(DictLearnTTAProgram):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.result_dir = self.result_dir / "layer_wise_dict"
        if cfg.version is not None:
            self.result_dir = self.result_dir / f"version_{cfg.version}"
        self.result_dir.mkdir(parents=True, exist_ok=True)
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
        for layer_idx, param_key in enumerate(self.task_vectors[0].keys()):
            model_task_vectors[param_key] = 0
            for task_idx in range(self.num_tasks):
                model_task_vectors[param_key] += (
                    coding[task_idx, layer_idx] * self.task_vectors[task_idx][param_key]
                )
        return model_task_vectors

    def load_models(self):
        cfg = self.cfg
        self.num_tasks = len(cfg.test_datasets)

        # load flan-t5 models
        super(DictLearnTTAProgram, self).load_models(cfg.task_vector_device)
        self.num_layers = len(self.task_vectors[0])

        self.load_dict_models(
            coding_size=self.num_tasks * self.num_layers
        )  # NOTE: coding_size = self.num_tasks * self.num_layers if layer-wise codings

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

    def model_forward_logits(self, x, *, task_idx: int, task_name: str):
        cfg = self.cfg
        input_ids, attention_mask = x

        max_len = input_ids.size(1)
        while torch.all(attention_mask[:, max_len - 1] == 0):
            max_len -= 1
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]

        # compute the dictionary codings
        dict_features = self.dict_feature_extractor(input_ids)
        dict_codings = self.dict_mapping(dict_features).to(cfg.task_vector_device)

        # compute the model logits
        model_logits = []

        # * NOTE: to reduce the memory usage, we compute the model features batch by batch in Program2
        dict_codings = dict_codings.mean(dim=0, keepdim=True)
        model_task_vector = self.compute_task_vectors(dict_codings, 0)
        model_task_vector = {
            k: v.to(self.forward_device(task_idx)) for k, v in model_task_vector.items()
        }
        model_sd = state_dict_add(
            self.pretrained_sd_on_device(self.forward_device(task_idx)),
            model_task_vector,
            strict=False,
        )
        forward_model = self.forward_model_on_device(self.forward_device(task_idx))
        forward_model.train()

        model_logits = (
            functional_call(
                forward_model,
                model_sd,
                args=(),
                kwargs=dict(
                    input_ids=input_ids.to(self.forward_device(task_idx)),
                    attention_mask=attention_mask.to(self.forward_device(task_idx)),
                    decoder_input_ids=torch.ones(
                        input_ids.size(0),  # mini batch size
                        1,
                        dtype=torch.long,
                        device=self.forward_device(task_idx),
                    )
                    * self.tokenizer.pad_token_id,
                ),
                tie_weights=False,
                strict=False,
            )
            .logits[:, 0, :]
            .to(self.forward_device(0), non_blocking=True)
        )

        return model_logits  # on the first forward device


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="flan_t5_dict",
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
