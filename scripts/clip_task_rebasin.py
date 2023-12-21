from _common import *

log = logging.getLogger(__name__)

from clip_checkpoint_path import (
    CHECKPOINT_DIR,
    finetuned_model_path,
    pretrained_model_path,
)

from src.clip_eval import eval_single_dataset_preprocess_head
from src.heads import get_classification_head
from src.modeling import ImageEncoder


class Program:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        cfg.save = str(CHECKPOINT_DIR / cfg.model)

    def run(self):
        self.load_models()

        self.eval_task_rebasin()

    def eval_task_rebasin(self):
        pass

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


@hydra.main(config_path=str(CONFIG_DIR), config_name="clip_default", version_base=None)
def main(cfg: DictConfig) -> None:
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
