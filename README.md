# MSTIL: Multi-cue Shape-aware Transferable Imbalance Learning for Effective Graphic API Recommendation

The code of MSTIL: Multi-cue Shape-aware Transferable Imbalance Learning for Effective Graphic API Recommendation.

The Module is a required package for calling EfficientNet-b3.

Datasets and other resources are available at https://github.com/cqu-isse/Plot2API.

To train LORA on SER30K on a single node with 2 gpus for 50 epochs run:
```shell
python -m torch.distributed.launch --nproc_per_node=2 --master_port=6666 \
--use_env main.py \
--config configs/pvt/pvt_small.py \
--visfinetune weights/pvt_small.pth \
--output_dir checkpoints/SER \
--dataset SER \
--data-path {path to SER30K dataset} \
--alpha 8 \
--batch-size 16 \
--locals 1 1 1 0
```
