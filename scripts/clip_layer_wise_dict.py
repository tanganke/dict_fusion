from _common import *

log = logging.getLogger(__name__)

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
from src.clip_eval import eval_single_dataset, eval_single_dataset_preprocess_head
from src.datasets.common import maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from src.tasks import check_parameterNamesMatch
from src.tasks.arithmetic import *
from src.tasks.task_vector import StateDict, TaskVector
from src.ties_merging_utils import state_dict_to_vector, vector_to_state_dict
from src.utils import timeit_context


class Program:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(DATA_DIR)

        self.result_dir = RESULTS_DIR / cfg.model / "layer_wise_dict"
        if cfg.version is not None:
            self.result_dir = self.result_dir / f"version_{cfg.version}"
        self.result_dir.mkdir(exist_ok=True, parents=True)
        self.result_path = self.result_dir / "results.csv"

    def run(self):
        self.load_models()
        self.load_datasets()

        if self.cfg.eval_dict_tta:
            self.eval_dict_tta()
        if self.cfg.eval_dict:
            self.eval_dict()

    @property
    @torch.no_grad()
    def task_vectors(self):
        with timeit_context("Computing task vectors"):
            task_vectors_as_dict = [
                state_dict_sub(
                    ft.state_dict(),
                    self.pretrained_model.state_dict(),
                    strict=False,
                )
                for ft in self.finetuned_models
            ]
            self._reference_sd = task_vectors_as_dict[0]
            task_vectors = torch.vstack(
                [state_dict_to_vector(d) for d in task_vectors_as_dict]
            )
            task_vectors = task_vectors
        return task_vectors

    def _sd_from_task_vector(self, task_vector):
        task_vector_as_dict = vector_to_state_dict(task_vector, self._reference_sd)
        sd = state_dict_add(task_vector_as_dict, self.pretrained_sd, strict=False)
        return sd

    def eval_dict_tta(self):
        with timeit_context("Computing task vectors"):
            task_vectors_as_dict = [
                state_dict_sub(
                    ft.state_dict(),
                    self.pretrained_model.state_dict(),
                    strict=False,
                    device=self.cfg.dict_mapping_device,
                )
                for ft in self.finetuned_models
            ]

        optimizer = torch.optim.Adam(self.dict_mapping.parameters(), lr=self.cfg.lr)
        self.dict_mapping.train()
        for step_idx in tqdm(range(1000), "training dict mapping"):
            losses = 0
            for datasset_idx, dataset_name in enumerate(self.cfg.test_datasets):
                batch = next(self.shuffled_test_loader_iters[datasset_idx])
                batch = maybe_dictionarize(batch)
                x = batch["images"]  # use images only

                dict_input = self.dict_preprocess(
                    x, return_tensors="pt"
                ).pixel_values.to(self.cfg.dict_mapping_device, non_blocking=True)
                dict_features = self.dict_feature_extractor(dict_input)
                dict_code = self.dict_mapping(dict_features)

                model_input = self.model_preprocess(x).to("cuda", non_blocking=True)
                model_features = []
                for i in range(x.size(0)):
                    model_task_vectors = {}
                    code = dict_code[i].view(self.num_tasks, self.num_layers)
                    for j, k in enumerate(task_vectors_as_dict[0].keys()):
                        model_task_vectors[k] = 0
                        for l in range(self.num_tasks):
                            model_task_vectors[k] += (
                                code[l, j] * task_vectors_as_dict[l][k]
                            )
                    model_sd = state_dict_add(
                        self.pretrained_sd, model_task_vectors, device="cuda"
                    )
                    _model_feature = functional_call(
                        self.forward_model,
                        model_sd,
                        model_input[i : i + 1],
                    )
                    model_features.append(_model_feature)
                model_features = torch.cat(model_features)
                model_logits = self.classification_heads[dataset_name](model_features)
                loss = softmax_entropy(model_logits).mean(0)
                losses += loss

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if (step_idx + 1) % 50 == 0:
                (self.result_dir / "checkpoints").mkdir(exist_ok=True)
                torch.save(
                    self.dict_mapping,
                    self.result_dir
                    / "checkpoints"
                    / f"dict_mapping_step={step_idx+1}.pt",
                )

    @torch.inference_mode()
    def eval_dict(self):
        with timeit_context("Computing task vectors"):
            task_vectors_as_dict = [
                state_dict_sub(
                    ft.state_dict(),
                    self.pretrained_model.state_dict(),
                    strict=False,
                    device="cuda",
                )
                for ft in self.finetuned_models
            ]
        results = defaultdict(list)

        for step_idx in tqdm(
            range(100, 1001, 100), "evaluating dict mapping", leave=False
        ):
            self.dict_mapping = torch.load(
                self.result_dir / "checkpoints" / f"dict_mapping_step={step_idx}.pt"
            )
            self.dict_mapping.eval()
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

                    dict_input = self.dict_preprocess(
                        x, return_tensors="pt"
                    ).pixel_values.to(self.cfg.dict_mapping_device, non_blocking=True)
                    dict_features = self.dict_feature_extractor(dict_input)
                    dict_code = self.dict_mapping(dict_features)

                    model_input = self.model_preprocess(x).to("cuda", non_blocking=True)
                    model_features = []
                    for i in range(x.size(0)):
                        model_task_vectors = {}
                        code = dict_code[i].view(self.num_tasks, self.num_layers)
                        for j, k in enumerate(task_vectors_as_dict[0].keys()):
                            model_task_vectors[k] = 0
                            for l in range(self.num_tasks):
                                model_task_vectors[k] += (
                                    code[l, j] * task_vectors_as_dict[l][k]
                                )
                        model_sd = state_dict_add(
                            self.pretrained_sd, model_task_vectors
                        )
                        _model_feature = functional_call(
                            self.forward_model,
                            model_sd,
                            model_input[i : i + 1],
                        )
                        model_features.append(_model_feature)
                    model_features = torch.cat(model_features)
                    model_logits = self.classification_heads[dataset_name](
                        model_features
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

    def eval_individuals(self):
        results = defaultdict(list)

        self.model = self.pretrained_model
        _result = defaultdict(list)
        self.eval_model_on_datasets(epoch_idx=0, results=_result)
        results["model"].append("pretrained")
        for dataset_name, acc in zip(_result["dataset"], _result["acc"]):
            results[dataset_name].append(acc)
        print(df := pd.DataFrame(results))
        df.to_csv(self.result_path, index=False)

        for dataset_name, image_encoder in track(
            zip(self.cfg.datasets, self.finetuned_models),
            "evaluating finetuned models",
        ):
            self.model = image_encoder
            _result = defaultdict(list)
            self.eval_model_on_datasets(epoch_idx=0, results=_result)
            results["model"].append(dataset_name)
            for dataset_name, acc in zip(_result["dataset"], _result["acc"]):
                results[dataset_name].append(acc)
        print(df := pd.DataFrame(results))
        df.to_csv(self.result_path, index=False)

    def load_models(self):
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
            dataset_name: get_classification_head(cfg, dataset_name).cuda()
            for dataset_name in cfg.test_datasets
        }

        self.model_preprocess = torchvision.transforms.Compose(
            self.pretrained_model.val_preprocess.transforms[
                -1:
            ]  # only normalization left, see `self.load_datasets`
        )
        self.forward_model = deepcopy(self.pretrained_model).to("cuda")
        for p in self.forward_model.parameters():
            p.requires_grad = False
        self.forward_model.eval()
        self.pretrained_sd = self.pretrained_model.state_dict()
        self.pretrained_sd = {
            k: v.to(cfg.dict_mapping_device, non_blocking=True)
            for k, v in self.pretrained_sd.items()
        }

        self.num_tasks = len(cfg.test_datasets)

        # load dict feature extractor model
        self.dict_preprocess = AutoFeatureExtractor.from_pretrained(
            cfg.dict_feature_extractor
        )
        dict_feature_extractor = ResNetForImageClassification.from_pretrained(
            cfg.dict_feature_extractor
        )
        dict_feature_extractor.classifier = torch.nn.Flatten(1, -1)
        self._dict_feature_extractor = dict_feature_extractor.to(
            cfg.dict_mapping_device, non_blocking=True
        )
        for p in self._dict_feature_extractor.parameters():
            p.requires_grad = False
        self._dict_feature_extractor.eval()
        self.dict_feature_extractor = lambda pixel_values: self._dict_feature_extractor(
            pixel_values=pixel_values
        ).logits.to(
            cfg.dict_mapping_device
        )  # in fact, this is the extracted feature, not logits

        # dict mapping
        if False:  # single layer
            self.dict_mapping = torch.nn.Linear(
                dict_feature_extractor.config.hidden_sizes[-1], self.num_tasks
            ).to(cfg.dict_mapping_device, non_blocking=True)
            self.dict_mapping.weight.data.zero_()
            self.dict_mapping.bias.data.fill_(0.3)
        else:
            task_vector_0 = state_dict_sub(
                finetuned_models[0].state_dict(),
                pretrained_model.state_dict(),
                strict=False,
            )
            self.num_layers = len(task_vector_0)
            _dict_mapping = torch.nn.Linear(
                dict_feature_extractor.config.hidden_sizes[-1],
                self.num_tasks * self.num_layers,
            )
            _dict_mapping.weight.data.zero_()
            _dict_mapping.bias.data.fill_(0.3)
            self.dict_mapping = torch.nn.Sequential(
                torch.nn.Linear(
                    dict_feature_extractor.config.hidden_sizes[-1],
                    dict_feature_extractor.config.hidden_sizes[-1],
                ),
                torch.nn.ReLU(),
                _dict_mapping,
            ).to(cfg.dict_mapping_device, non_blocking=True)

    def load_datasets(self):
        import open_clip.transform
        import torchvision.transforms

        from src.datasets.registry import get_dataset

        cfg = self.cfg

        datasets = [
            get_dataset(
                dataset_name,
                torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize((224, 224)),
                        open_clip.transform._convert_to_rgb,
                        torchvision.transforms.ToTensor(),
                    ]
                ),
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
                pin_memory=True,
            )
            for d in self.test_datasets
        ]
        self.shuffled_test_loader_iters = [
            iter(itertools.cycle(d)) for d in self.shuffled_test_loaders
        ]


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="clip_dict",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
