copyed dictmapping from ViT-B-32 task_wise_dict version 1

## Evaluation

```bash
python scripts/clip_task_wise_dict.py batch_size=16 version=1 \
        forward_devices=cuda task_vector_device=cuda \
        eval_dict_tta=false eval_dict=true \
        dict_feature_extractor=microsoft/resnet-50 \
        model=ViT-L-14
```
