## Test-time adaptation

```bash
python scripts/clip_task_wise_dict.py version=3 \
    batch_size=16 \
    eval_dict_tta=true eval_dict=false \
    dict_feature_extractor=microsoft/resnet-50 \
    forward_devices=\[1,2,3,4\] \
    model=ViT-L-14
```

## Evaluation

```bash
python scripts/clip_task_wise_dict.py version=3 \
    batch_size=16 \
    eval_dict_tta=false eval_dict=true \
    dict_feature_extractor=microsoft/resnet-50 \
    model=ViT-L-14
```