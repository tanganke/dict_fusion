## Test-time Adaptation

```bash
python scripts/flan_t5_task_wise_dict.py version=3 \
    batch_size=8 lr=5e-6 \
    dict_feature_extractor=msmarco-MiniLM-L12-cos-v5 \
    eval_dict_tta=true eval_dict=false
```

## Evaluation

```bash
python scripts/flan_t5_task_wise_dict.py version=3 \
    batch_size=8 lr=5e-6 \
    dict_feature_extractor=msmarco-MiniLM-L12-cos-v5 \
    eval_dict_tta=false eval_dict=true
```
