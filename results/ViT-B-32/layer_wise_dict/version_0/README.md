layer-wise, resnet-18 as feature extractor

## Test-time adaptation

```bash
python scripts/clip_layer_wise_dict.py version=0 \
    batch_size=16 \
    eval_dict_tta=true eval_dict=false
```

## Evaluation

```bash
python scripts/clip_layer_wise_dict.py version=0 \
    batch_size=16 \
    eval_dict_tta=false eval_dict=true
```
