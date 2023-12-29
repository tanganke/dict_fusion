zero-shot transfer experiments

copy from version 3 flan-t5-base

## Evaluation

```bash
python scripts/flan_t5_task_wise_dict.py version=0 \
    batch_size=8 lr=5e-6 \
    dict_feature_extractor=msmarco-MiniLM-L12-cos-v5 \
    eval_dict_tta=false eval_dict=true \
    models=flan-t5-large
```