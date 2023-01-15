# MSTIL: Multi-cue Shape-aware Transferable Imbalance Learning for Effective Graphic API Recommendation

# The code of MSTIL: Multi-cue Shape-aware Transferable Imbalance Learning for Effective Graphic API Recommendation.

# The required resources

The Module is a required package for calling EfficientNet-b3.

Datasets and other resources are available at https://pan.baidu.com/s/1I8btvuLwn5w3GnI-ZCV3Ew (the code is ISSE).

# Train and Test

To train and test the model with MSTIL (resnet for example):
```shell
python resnet.py -g[the id of your gpu]
```
Models | Caffe | Keras | TensorFlow | CNTK | MXNet | PyTorch  | CoreML | ONNX
:-----:|:-----:|:-----:|:----------:|:----:|:-----:|:--------:|:------:|:-----:|
[VGG 19](https://arxiv.org/abs/1409.1556.pdf) | √ | √ | √ | √ | √ | √ | √ | √
[Inception V1](https://arxiv.org/abs/1409.4842v1) | √ | √ | √ | √ | √ | √ | √ | √
[Inception V3](https://arxiv.org/abs/1512.00567)  | √ | √ | √ | √ | √ | √ | √ | √
[Inception V4](https://arxiv.org/abs/1512.00567)  | √ | √ | √ | o | √ | √ | √ | √
[ResNet V1](https://arxiv.org/abs/1512.03385)                               |   ×   |   √   |     √      |   o  |   √   |    √ | √ | √
[ResNet V2](https://arxiv.org/abs/1603.05027)                               |   √   |   √   |     √      |   √  |   √   | √ | √ | √
[MobileNet V1](https://arxiv.org/pdf/1704.04861.pdf)                        |   ×   |   √   |     √      |   o  |   √   |    √       | √ | √ | √
[MobileNet V2](https://arxiv.org/pdf/1704.04861.pdf)                        |   ×   |   √   |     √      |   o  |   √   |    √       | √ | √ | √
[Xception](https://arxiv.org/pdf/1610.02357.pdf)                            |   √   |   √   |     √      |   o  |   ×   |    √ | √ | √ | √
[SqueezeNet](https://arxiv.org/pdf/1602.07360)                              |   √   |   √   |     √      |   √  |   √   |    √ | √ | √ | √
[DenseNet](https://arxiv.org/abs/1608.06993)                                |   √   |   √   |     √      |   √  |   √   |    √       | √ | √
[NASNet](https://arxiv.org/abs/1707.07012)                                  |   x   |   √   |     √      |   o  |   √   | √ | √ | x
[ResNext](https://arxiv.org/abs/1611.05431)                                 |   √   |   √   |     √      |   √  |   √   | √ | √ | √ | √ | √
[voc FCN](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) |       |       |     √      |   √  |       |

**If you find this code to be useful for your research, please consider citing.**
