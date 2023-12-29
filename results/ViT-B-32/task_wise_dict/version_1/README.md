task-wise, resnet-50 as feature extractor

## Test-time adaptation

```bash
python scripts/clip_task_wise_dict.py version=1 \
        batch_size=16 \
        eval_dict_tta=true eval_dict=false \
        dict_feature_extractor=microsoft/resnet-50
```

## Evaluation

```bash
python scripts/clip_task_wise_dict.py batch_size=16 version=1 \
        forward_devices=cuda task_vector_device=cuda \
        eval_dict_tta=false eval_dict=true \
        dict_feature_extractor=microsoft/resnet-50
```
