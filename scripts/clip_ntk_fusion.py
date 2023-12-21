from _common import *

log = logging.getLogger(__name__)

from collections import defaultdict

from clip_checkpoint_path import (
    CHECKPOINT_DIR,
    finetuned_model_path,
    pretrained_model_path,
)

from src.clip_eval import eval_single_dataset_preprocess_head
from src.datasets.common import maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from src.tasks.arithmetic import *
from src.ties_merging_utils import state_dict_to_vector, vector_to_state_dict


class Program:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(DATA_DIR)

        results_dir = RESULTS_DIR / cfg.model
        results_dir.mkdir(exist_ok=True, parents=True)
        results_path = results_dir / "clip_ntk_fusion.csv"

        self.results_dir = results_dir
        self.results_path = results_path

    def run(self):
        self.load_models()
        self.load_datasets()

        self.eval_nkt_fusion()

    @property
    @torch.no_grad()
    def finetuned_models_cuda(self):
        if hasattr(self, "_finetuned_models"):
            return self._finetuned_models_cuda
        else:
            self._finetuned_models_cuda = [
                model.to("cuda", non_blocking=True) for model in self.finetuned_models
            ]
            return self._finetuned_models_cuda

    def eval_nkt_fusion(self):
        with timeit_context("Computing task vectors"):
            task_vectors_as_dict = [
                state_dict_sub(
                    ft.state_dict(), self.pretrained_model.state_dict(), strict=False
                )
                for ft in self.finetuned_models
            ]
            task_vectors = torch.vstack(
                [state_dict_to_vector(d) for d in task_vectors_as_dict]
            )

        results = defaultdict(list)

        num_tasks = task_vectors.size(0)

        global_model = deepcopy(self.pretrained_model)
        global_task_vector_as_dict = state_dict_avg(task_vectors_as_dict)
        global_state_dict = state_dict_add(
            global_model.state_dict(), global_task_vector_as_dict, strict=False
        )
        global_model.load_state_dict(global_state_dict, strict=False)
        global_model = global_model.to("cuda", non_blocking=True)
        for epoch_idx in tqdm(range(1, 30), "epochs", leave=False):
            lambda_c = torch.empty(1, num_tasks, device="cuda", dtype=torch.float32)

            x = []
            for dataset_idx, dataset_name in enumerate(self.cfg.test_datasets):
                batch = next(iter(self.test_dataloaders[dataset_idx]))
                batch = maybe_dictionarize(batch)
                _x = batch["images"].to("cuda", non_blocking=True)
                x.append(_x)
            x = torch.cat(x)

            for dataset_idx, dataset_name in enumerate(
                tqdm(
                    self.cfg.test_datasets,
                    "computing task-wise weight",
                    leave=False,
                )
            ):
                with torch.no_grad():
                    global_features = global_model(x)

                with torch.no_grad():
                    local_features = self.finetuned_models_cuda[dataset_idx](x)

                global_sigma = global_features @ global_features.T
                local_sigma = local_features @ local_features.T

                _lambda_c = global_sigma.trace() / (
                    local_sigma.trace() + global_sigma.trace()
                )
                lambda_c[0, dataset_idx] = _lambda_c

            lambda_c = lambda_c.cpu().numpy()[0]
            log.info(f"lambda_c: {lambda_c}")
            global_model = deepcopy(self.pretrained_model)
            global_task_vector_as_dict = state_dict_weighted_sum(
                task_vectors_as_dict, lambda_c
            )
            global_state_dict = state_dict_add(
                global_model.state_dict(), global_task_vector_as_dict, strict=False
            )
            global_model.load_state_dict(global_state_dict, strict=False)
            global_model = global_model.to("cuda", non_blocking=True)
            _results = self.eval_model_on_datasets(global_model, self.cfg.test_datasets)
            results["epoch"] = epoch_idx
            for dataset_name, acc in _results.items():
                results[dataset_name].append(acc)

            print(df := pd.DataFrame(results))
            df.to_csv(self.results_path, index=False)

    def load_models(self):
        cfg = self.cfg
        with timeit_context("Loading pretrained model"):
            self.pretrained_model = torch.load(
                pretrained_model_path(cfg.model), map_location="cpu"
            )

        with timeit_context("Loading finetuned models"):
            self.finetuned_models = [
                torch.load(
                    finetuned_model_path(cfg.model, dataset_name), map_location="cpu"
                )
                for dataset_name in track(cfg.test_datasets, "loading finetuned models")
            ]

        self.classification_heads = {
            dataset_name: get_classification_head(cfg, dataset_name)
            for dataset_name in cfg.test_datasets
        }

    def load_datasets(self):
        cfg = self.cfg
        from src.datasets.common import get_dataloader, maybe_dictionarize
        from src.datasets.registry import get_dataset

        train_datasets = [
            get_dataset(
                dataset_name,
                self.pretrained_model.val_preprocess,
                location=cfg.data_location,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )
            for dataset_name in cfg.test_datasets
        ]
        train_dataloaders = [
            get_dataloader(dataset, is_train=True, args=cfg, image_encoder=None)
            for dataset in train_datasets
        ]

        self.train_dataloaders = train_dataloaders
        self.test_dataloaders = [
            get_dataloader(dataset, is_train=False, args=cfg, image_encoder=None)
            for dataset in train_datasets
        ]

    def eval_model_on_datasets(
        self,
        model: ImageEncoder,
        test_datasets: List[str],
    ):
        results = {}

        Total_ACC = 0
        for dataset_name in tqdm(
            test_datasets,
            desc="Evaluating on datasets",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            leave=False,
        ):
            classification_head = self.classification_heads[dataset_name]
            metrics = eval_single_dataset_preprocess_head(
                model, classification_head, dataset_name, self.cfg
            )
            Total_ACC += (acc := metrics["top1"])
            log.info(
                "Eval: init: "
                + " dataset: "
                + str(dataset_name)
                + " ACC: "
                + str(metrics["top1"])
            )

            results[dataset_name] = acc

        log.info(
            "Eval: init: " + " Avg ACC:" + str(Total_ACC / len(test_datasets)) + "\n"
        )

        return results


@hydra.main(config_path=str(CONFIG_DIR), config_name="clip_default", version_base=None)
def main(cfg: DictConfig) -> None:
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
