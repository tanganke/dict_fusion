from _common import *
from torch._C import device

log = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from collections import defaultdict

import flan_t5_individuals
from torch.func import functional_call

from src.adamerging import softmax_entropy
from src.tasks.arithmetic import *
from src.utils import num_parameters


class DictLearnTTAProgram(flan_t5_individuals.Program, ABC):
    @abstractmethod
    def compute_task_vectors(
        self,
        dict_codings: Tensor,
        sample_idx: int,
    ):
        ...

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        if hasattr(cfg, "seed") and cfg.seed is not None:
            log.info(f"set seed to {cfg.seed}")
            L.seed_everything(cfg.seed)

        if cfg.peft.peft_config is None:
            self.result_dir = RESULTS_DIR / cfg.model.name
        else:
            self.result_dir = RESULTS_DIR / (cfg.model.name + "_" + cfg.peft.name)

    def run(self):
        self.load_models()

        if self.cfg.profile:
            self.profile()

        if not (self.cfg.eval_dict_tta or self.cfg.eval_dict):
            return

        self.load_datasets(setup_dataloaders=False)

        if self.cfg.eval_dict_tta:
            self.eval_dict_tta()
        if self.cfg.eval_dict:
            self.eval_dict()

    def profile(self):
        """
        list the number of parameters of self._feature_extractor and self._dict_mapping and self.forward_model
        """
        print("feature extractor")
        print(num_parameters(self._dict_feature_extractor))
        print("dict mapping")
        print(num_parameters(self._dict_mapping))
        print("forward model")
        print(num_parameters(self.forward_model))
        if hasattr(self, "num_layers"):
            print("num layers")
            print(self.num_layers)

    def eval_dict_tta(self):
        optimizer = torch.optim.Adam(self._dict_mapping.parameters(), lr=self.cfg.lr)
        self._dict_mapping.train()
        for step_idx in tqdm(range(1000), "training dict mapping"):
            losses = 0
            for dataset_idx, dataset_name in enumerate(self.cfg.test_datasets):
                batch = next(self.shuffled_test_loader_iters[dataset_name])
                x = (
                    input_ids := batch["input_ids"],
                    attention_mask := batch["attention_mask"],
                )  #! use inputs only, not labels

                model_logits = self.model_forward_logits(
                    x, task_idx=dataset_idx, task_name=dataset_name
                )
                loss = softmax_entropy(model_logits).mean(0)
                losses += loss

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if (step_idx + 1) % 50 == 0:
                (self.result_dir / "checkpoints").mkdir(exist_ok=True)
                torch.save(
                    self._dict_mapping,
                    self.result_dir
                    / "checkpoints"
                    / f"dict_mapping_step={step_idx+1}.pt",
                )

    @torch.inference_mode()
    def eval_dict(self):
        results = defaultdict(list)

        for step_idx in tqdm(
            reversed(range(100, 1001, 100)), "evaluating dict mapping", leave=False
        ):
            self._dict_mapping = torch.load(
                self.result_dir / "checkpoints" / f"dict_mapping_step={step_idx}.pt",
                map_location=torch.device(self.cfg.dict_mapping_device),
            )
            self._dict_mapping.eval()
            results["step"].append(step_idx)

            for dataset_idx, dataset_name in enumerate(
                tqdm(
                    self.cfg.test_datasets,
                    "evaluating datasets",
                    leave=False,
                )
            ):
                log.info(f"evaluating {dataset_name}")
                test_loader = self.test_loaders[dataset_idx]
                if self.cfg.fast_dev_run:
                    log.info("fast_dev_run: only use the first batch")
                    test_loader = [next(iter(test_loader))]
                score = flan_t5_individuals.metric_func[dataset_name](
                    self, test_loader, self.tokenizer
                )
                results[dataset_name].append(score)
                log.info(f"{dataset_name}: {score:.3f}")
            (df := pd.DataFrame(results)).to_csv(self.result_path, index=False)
            print(df)

    @functools.cache
    def forward_device(self, sample_idx: int):
        cfg = self.cfg

        if not isinstance(cfg.forward_devices, (list, ListConfig)):
            return torch.device(cfg.forward_devices)
        else:
            num_forward_devices = len(cfg.forward_devices)
            assert (
                cfg.batch_size % num_forward_devices == 0
            ), "batch size must be divisible by the number of forward devices"
            forward_device_idx = sample_idx // (cfg.batch_size // num_forward_devices)
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
        for sample_idx in range(input_ids.size(0)):
            model_task_vector = self.compute_task_vectors(dict_codings, sample_idx)
            model_task_vector = {
                k: v.to(self.forward_device(sample_idx))
                for k, v in model_task_vector.items()
            }
            model_sd = state_dict_add(
                self.pretrained_sd_on_device(self.forward_device(sample_idx)),
                model_task_vector,
                strict=False,
            )

            forward_model = self.forward_model_on_device(
                self.forward_device(sample_idx)
            )
            _model_logits = (
                functional_call(
                    forward_model,
                    model_sd,
                    args=(),
                    kwargs=dict(
                        input_ids=input_ids[sample_idx : sample_idx + 1].to(
                            self.forward_device(sample_idx)
                        ),
                        attention_mask=attention_mask[sample_idx : sample_idx + 1].to(
                            self.forward_device(sample_idx)
                        ),
                        decoder_input_ids=torch.ones(
                            1,  # mini batch size
                            1,
                            dtype=torch.long,
                            device=self.forward_device(sample_idx),
                        )
                        * self.tokenizer.pad_token_id,
                    ),
                    tie_weights=False,
                    strict=False,
                )
                .logits[:, 0, :]
                .to(self.forward_device(0), non_blocking=True)
            )

            model_logits.append(_model_logits)
        model_logits = torch.cat(model_logits)
        return model_logits  # on the first forward device

    def eval(self):
        self.forward_model.eval()
        return self

    def generate(self, input_ids: Tensor):
        """
        To mimic a AutoModelForSeq2SeqLM, we need to implement a generate method.
        """
        cfg = self.cfg

        # compute the dictionary codings
        dict_features = self.dict_feature_extractor(input_ids)
        dict_codings = self.dict_mapping(dict_features).to(cfg.task_vector_device)

        # compute the model logits
        model_output = []
        for sample_idx in range(input_ids.size(0)):
            model_task_vector = self.compute_task_vectors(dict_codings, sample_idx)
            model_task_vector = {
                k: v.to(self.forward_device(sample_idx))
                for k, v in model_task_vector.items()
            }
            model_sd = state_dict_add(
                self.pretrained_sd_on_device(self.forward_device(sample_idx)),
                model_task_vector,
                strict=False,
            )
            self.forward_model.load_state_dict(
                self.pretrained_sd_on_device(self.forward_device(sample_idx)),
                strict=True,
                assign=True,
            )  #  load the original weights
            self.forward_model.load_state_dict(
                model_sd,
                strict=False,
                assign=True,
            )  # update modified weights
            _output = self.forward_model.generate(
                input_ids[sample_idx : sample_idx + 1].to(
                    self.forward_device(sample_idx)
                )
            ).to(self.forward_device(0), non_blocking=True)
            model_output.append(_output)
        max_len = max([o.size(1) for o in model_output])
        for i in range(len(model_output)):
            # padding to the same length
            model_output[i] = torch.nn.functional.pad(
                model_output[i],
                (0, max_len - model_output[i].size(1)),
                value=self.tokenizer.pad_token_id,
            )
        model_output = torch.cat(model_output)
        return model_output  # on the first forward device

    def load_dict_models(self, coding_size: int):
        from sentence_transformers import SentenceTransformer

        cfg = self.cfg

        self._dict_feature_extractor = dict_feature_extractor = SentenceTransformer(
            cfg.dict_feature_extractor,
            device=torch.device(cfg.dict_feature_extractor_device),
        ).eval()

        # * dict mapping
        # dict mapping is a two-layer MLP
        # this last layer is initialized so that the initial outputs are always 0.3
        _dict_mapping = torch.nn.Linear(
            dict_feature_extractor.get_sentence_embedding_dimension(), coding_size
        )
        _dict_mapping.weight.data.zero_()
        _dict_mapping.bias.data.fill_(0.3)
        self._dict_mapping = torch.nn.Sequential(
            torch.nn.Linear(
                dict_feature_extractor.get_sentence_embedding_dimension(),
                dict_feature_extractor.get_sentence_embedding_dimension(),
            ),
            torch.nn.ReLU(),
            _dict_mapping,
        ).to(cfg.dict_mapping_device, non_blocking=True)

    @torch.no_grad()
    def dict_feature_extractor(self, input_ids: Tensor):
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        return self._dict_feature_extractor.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        ).to(self.cfg.dict_mapping_device)

    def dict_mapping(self, features: Tensor) -> Tensor:
        features = features.to(self.cfg.dict_mapping_device)
        return self._dict_mapping(features)

    @functools.cache
    def pretrained_sd_on_device(self, device):
        return {
            k: v.to(device, non_blocking=True) for k, v in self.pretrained_sd.items()
        }

    def load_models(self):
        cfg = self.cfg
        self.num_tasks = len(cfg.test_datasets)

        # load flan-t5 models
        super().load_models(cfg.task_vector_device)
        self.num_layers = len(self.task_vectors[0])

        self.load_dict_models(
            coding_size=self.num_tasks
        )  # NOTE: coding_size = self.num_tasks * self.num_layers if layer-wise codings

        # setup forward model, this is used to perform inference
        self.forward_model = self.forward_model_on_device("cpu")
        self.pretrained_sd = self.pretrained_model.state_dict()

    @functools.cache
    def forward_model_on_device(self, device):
        return deepcopy(self.pretrained_model).to(device)
