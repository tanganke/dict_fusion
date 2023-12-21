from _common import *

log = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from collections import defaultdict

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
from src.datasets.common import maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ImageEncoder
from src.tasks.arithmetic import *
from src.utils import timeit_context


class DictLearnTTAProgram(ABC):
    @abstractmethod
    def compute_task_vectors(
        self,
        dict_codings: Tensor,
        sample_idx: int,
    ):
        ...

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(DATA_DIR)

    def run(self):
        self.load_models()
        self.load_datasets()

        if self.cfg.eval_dict_tta:
            self.eval_dict_tta()
        if self.cfg.eval_dict:
            self.eval_dict()

    def eval_dict_tta(self):
        optimizer = torch.optim.Adam(self._dict_mapping.parameters(), lr=self.cfg.lr)
        self._dict_mapping.train()
        for step_idx in tqdm(range(1000), "training dict mapping"):
            losses = 0
            for dataset_idx, dataset_name in enumerate(self.cfg.test_datasets):
                batch = next(self.shuffled_test_loader_iters[dataset_idx])
                batch = maybe_dictionarize(batch)
                x = batch["images"]  # use images only

                model_logits = self.model_forward(
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
                test_loader = self.test_loaders[dataset_idx]
                TOTAL_CORRECT = 0
                TOTAL_COUNT = 0
                for batch in (
                    pbar := tqdm(
                        test_loader,
                        f"evaluate {dataset_name}",
                        leave=False,
                    )
                ):
                    batch = maybe_dictionarize(batch)
                    x = batch["images"]
                    y = batch["labels"]

                    model_logits = self.model_forward(
                        x, task_idx=dataset_idx, task_name=dataset_name
                    )
                    model_preds = model_logits.argmax(-1).cpu()
                    correct = (model_preds == y).sum().item()
                    TOTAL_CORRECT += correct
                    TOTAL_COUNT += len(y)
                    acc = TOTAL_CORRECT / TOTAL_COUNT
                    pbar.set_postfix_str(f"acc={acc:.2f}")
                results[dataset_name].append(acc)
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

    def model_forward(self, x, *, task_idx: int, task_name: str):
        """
        Performs a forward pass through the model for the given input and dataset.

        It first preprocesses the input and moves it to the device specified in the configuration for dictionary mapping.
        It then extracts features from the preprocessed input using the dictionary feature extractor and converts the
        features to codings using the dictionary mapping.

        The method then preprocesses the input again and moves it to the device specified in the configuration for forward pass.
        It then performs a forward pass through the model for each sample in the input, using the task vectors and the
        pretrained state dict. The task vectors are weighted by the dictionary codings and added to the pretrained state dict.
        The forward pass is performed using the forward model and the updated state dict.
        The features from the forward pass are then passed through the classification head for the dataset to get the logits.

        Args:
            x: The inputs.
            task_idx (int): The index of the task.
            task_name (str): The name of the task. this is used to choose the classification head.

        Returns:
            Tensor: The logits for the input. on the first `cfg.forward_device`.

        Note:
            The `device` in this context refers to the hardware on which the computations are performed.
            It can be either a CPU or a GPU. The specific device is specified in the configuration and can be different
            for different parts of the computation. For example, the dictionary mapping is performed on the device specified
            in `cfg.dict_mapping_device`, while the forward pass is performed on the device specified in `cfg.forward_device`.
        """
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
        for sample_idx in range(x.size(0)):
            model_task_vector = self.compute_task_vectors(dict_codings, sample_idx)
            model_task_vector = {
                k: v.to(self.forward_device(sample_idx))
                for k, v in model_task_vector.items()
            }
            model_sd = state_dict_add(
                self.pretrained_sd_on_device(self.forward_device(sample_idx)),
                model_task_vector,
            )
            _model_feature = functional_call(
                self.forward_model,
                model_sd,
                model_input[sample_idx : sample_idx + 1].to(
                    self.forward_device(sample_idx)
                ),
            ).to(self.forward_device(0), non_blocking=True)
            model_features.append(_model_feature)
        model_features = torch.cat(model_features)
        model_logits = self.classification_heads[task_name](model_features)
        return model_logits  # on the first forward device

    def load_clip_models(self):
        """
        Loads the pretrained CLIP model and the fine-tuned models for each dataset specified in the configuration.
        It first loads the pretrained model from the path specified in the configuration.
        It then loads each fine-tuned model from the path specified in the configuration,
        using the name of the dataset to construct the path.
        Finally, it sets up the classification heads for each dataset, using the configuration and the name of the dataset.

        Side Effects:
            Sets the instance variables `pretrained_model`, `finetuned_models`, and `classification_heads`.
        """
        cfg = self.cfg

        # load pretrained and fine-tuned model
        with timeit_context():
            log.info("load models")
            pretrained_model = torch.load(
                pretrained_model_path(cfg.model), map_location="cpu"
            )
            finetuned_models = [
                torch.load(
                    finetuned_model_path(cfg.model, dataset_name), map_location="cpu"
                )
                for dataset_name in track(cfg.test_datasets, "loading finetuned models")
            ]

        self.pretrained_model: ImageEncoder = pretrained_model
        self.finetuned_models = finetuned_models
        self.classification_heads = {
            dataset_name: get_classification_head(cfg, dataset_name).to(
                self.forward_device(0), non_blocking=True
            )
            for dataset_name in cfg.test_datasets
        }

    def load_dict_models(self, coding_size: int):
        """
        Loads the dictionary models specified in the configuration. It first loads the dictionary feature extractor model
        from the path specified in the configuration and sets its classifier to a Flatten layer.
        It then moves the feature extractor to the device specified in the configuration and sets its parameters to not
        require gradients. It also sets the feature extractor to evaluation mode.
        The method then sets up a lambda function to extract features from pixel values using the feature extractor.
        Finally, it sets up the dictionary mapping as a two-layer MLP and moves it to the device specified in the configuration.

        Side Effects:
            Sets the instance variables `_dict_feature_extractor` and `_dict_mapping`.
        """
        cfg = self.cfg

        # load dict feature extractor model
        dict_feature_extractor = ResNetForImageClassification.from_pretrained(
            cfg.dict_feature_extractor
        )
        dict_feature_extractor.classifier = torch.nn.Flatten(
            1, -1
        )  # output the feature, not logits
        self._dict_feature_extractor = dict_feature_extractor.to(
            cfg.dict_feature_extractor_device, non_blocking=True
        )
        for p in self._dict_feature_extractor.parameters():
            p.requires_grad = False
        self._dict_feature_extractor.eval()

        # * dict mapping
        # dict mapping is a two-layer MLP
        # this last layer is initialized so that the initial outputs are always 0.3
        _dict_mapping = torch.nn.Linear(
            dict_feature_extractor.config.hidden_sizes[-1], coding_size
        )
        _dict_mapping.weight.data.zero_()
        _dict_mapping.bias.data.fill_(0.3)
        self._dict_mapping = torch.nn.Sequential(
            torch.nn.Linear(
                dict_feature_extractor.config.hidden_sizes[-1],
                dict_feature_extractor.config.hidden_sizes[-1],
            ),
            torch.nn.ReLU(),
            _dict_mapping,
        ).to(cfg.dict_mapping_device, non_blocking=True)

    def dict_feature_extractor(self, images: Tensor) -> Tensor:
        """
        Extracts features from the given images using the dictionary feature extractor.
        The extracted features are then moved to the device specified in the configuration.

        Args:
            images (Tensor): The images from which to extract features.

        Returns:
            Tensor: The extracted features.

        Note:
            Despite the name `logits`, the returned tensor actually represents the extracted features,
            not the logits. See `self.load_dict_models` for more details.
        """
        images = images.to(self.cfg.dict_feature_extractor_device)
        return self._dict_feature_extractor(pixel_values=images).logits.to(
            self.cfg.dict_mapping_device
        )  # in fact, this is the extracted feature, not logits, see `self.load_dict_models`

    def dict_mapping(self, features: Tensor) -> Tensor:
        """
        Converts the given features to codings using the dictionary mapping.
        The features are first moved to the device specified in the configuration and then passed through the dictionary mapping.

        Args:
            features (Tensor): The features to convert to codings.

        Returns:
            Tensor: The codings corresponding to the given features.
        """
        features = features.to(self.cfg.dict_mapping_device)
        return self._dict_mapping(features)

    def setup_preprocess(self):
        """
        Sets up the preprocessing for the model and the dictionary.

        Side Effects:
            Sets the instance variables `model_preprocess` and `dict_preprocess`.
        """
        cfg = self.cfg
        self.model_preprocess = torchvision.transforms.Compose(
            self.pretrained_model.val_preprocess.transforms[
                -1:
            ]  # only normalization left, see `self.load_datasets`
        )
        self.dict_preprocess = AutoFeatureExtractor.from_pretrained(
            cfg.dict_feature_extractor
        )

    @functools.cache
    def pretrained_sd_on_device(self, device):
        return {
            k: v.to(device, non_blocking=True) for k, v in self.pretrained_sd.items()
        }

    def load_models(self, *, free_finetuned_models: bool = True):
        cfg = self.cfg
        self.num_tasks = len(cfg.test_datasets)

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
            coding_size=self.num_tasks
        )  # NOTE: coding_size = self.num_tasks * self.num_layers if layer-wise codings
        self.setup_preprocess()

        # setup forward model, this is used to perform inference
        self.forward_model = deepcopy(self.pretrained_model)
        for p in self.forward_model.parameters():
            p.requires_grad = False
        self.forward_model.eval()
        self.pretrained_sd = self.pretrained_model.state_dict()

    def load_datasets(self):
        """
        Loads the datasets specified in the configuration.

        It first imports the necessary modules and sets up a basic transform for the images.
        It then loads each dataset specified in the configuration, applies the basic transform,
        and sets the location, batch size, and number of workers from the configuration.

        The test dataset from each loaded dataset is added to the list of test datasets.
        It then sets up the data loaders for the test datasets, both with
        and without shuffling, and creates an iterator for each shuffled test loader.

        Side Effects:
            Sets the instance variables `test_datasets`, `test_loaders`, `shuffled_test_loaders`, and
            `shuffled_test_loader_iters`.
        """
        import open_clip.transform
        import torchvision.transforms

        from src.datasets.registry import get_dataset

        cfg = self.cfg

        basic_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                open_clip.transform._convert_to_rgb,
                torchvision.transforms.ToTensor(),
            ]
        )

        datasets = [
            get_dataset(
                dataset_name,
                basic_transform,
                location=cfg.data_location,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )
            for dataset_name in cfg.test_datasets
        ]
        self.test_datasets = [d.test_dataset for d in datasets]
        self.test_loaders = [
            DataLoader(
                d,
                shuffle=False,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
            for d in self.test_datasets
        ]
        self.shuffled_test_loaders = [
            DataLoader(
                d,
                shuffle=True,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )
            for d in self.test_datasets
        ]
        self.shuffled_test_loader_iters = [
            iter(itertools.cycle(d)) for d in self.shuffled_test_loaders
        ]
