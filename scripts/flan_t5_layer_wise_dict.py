from typing import Any

from _common import *

from scripts._common import DictConfig

log = logging.getLogger(__name__)

from collections import defaultdict

import lightning as L
import lightning.fabric
import lightning.pytorch as pl
from flan_t5_checkpoint_path import finetuned_model_path
from flan_t5_individuals import Program as _Program
from flan_t5_individuals import metric_func
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator
from transformers.generation import GenerationConfig, GenerationMixin

from datasets import DatasetDict, load_dataset, load_from_disk
from src.adamerging import softmax_entropy
from src.tasks.arithmetic import state_dict_avg, state_dict_sub, state_dict_sum
from src.ties_merging_utils import *
from src.utils import num_devices, num_parameters, timeit_context

# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Program(_Program):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        if hasattr(cfg, "seed"):
            log.info(f"set seed to {cfg.seed}")
            L.seed_everything(cfg.seed)

        # setup results dir
        if cfg.peft.peft_config is None:
            self.results_dir = RESULTS_DIR / cfg.model.name
        else:
            self.results_dir = RESULTS_DIR / (cfg.model.name + "_" + cfg.peft.name)
        if cfg.version is not None:
            self.results_dir = self.results_dir / f"version_{cfg.version}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.fabric = L.Fabric(accelerator="cuda", devices=1)
        self.fabric.launch()

    def run(self):
        self.load_models(task_vector_device=torch.device("cuda:1"))
        self.load_datasets()
        self.initialize_merged_model()


@hydra.main(str(CONFIG_DIR), "flan_t5_default", None)
def main(cfg: DictConfig):
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
