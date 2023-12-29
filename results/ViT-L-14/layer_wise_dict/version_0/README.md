## Test-time adaptation

```bash
python scripts/clip_layer_wise_dict.py version=0 \
    batch_size=16 \
    eval_dict_tta=true eval_dict=false \
    forward_devices=\[1,2,3,4\] \
    model=ViT-L-14
```

## Evaluation

```bash
python scripts/clip_layer_wise_dict.py version=0 \
    batch_size=16 \
    eval_dict_tta=false eval_dict=true \
    model=ViT-L-14
```