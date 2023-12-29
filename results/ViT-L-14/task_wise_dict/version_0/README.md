copyed dictmapping from ViT-B-32 task_wise_dict version 0

## Evaluation

```bash
python scripts/clip_task_wise_dict.py batch_size=16 version=0 \
        forward_devices=cuda task_vector_device=cuda \
        eval_dict_tta=false eval_dict=true \
        model=ViT-L-14
```