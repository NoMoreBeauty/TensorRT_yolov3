<div align="center">

[简体中文](README.zh-CN.md)
<br>

[TensorRT](https://github.com/NVIDIA/TensorRT) 是一个有助于在NVIDIA图形处理单元（GPU）上高性能推理c++库。它旨在与TesnsorFlow、Caffe、Pytorch以及MXNet等训练框架以互补的方式进行工作，专门致力于在GPU上快速有效地进行网络推理。<br>

<span style="color: lightskyblue;">此项目将在DarkNet上训练的开源的yolov3模型首先转化为onnx模型，再使用TensorRT将onnx模型转化为TensorRT支持的trt模型，旨在对比模型转换前后的推理准确率、推理速度以及内存或显存占用并做比较</span>

<span style="color: lightskyblue;">对比的结果如下表所示：</span>
<div align="center">

| Class      | Images | Instances | Box(P | R | mAP50 | mAP50-95) |
|------------|--------|-----------|-------|---|-------|-----------|
| all        | 1497   | 7465      | 0.934 | 0.866 | 0.929 | 0.713 |
| Pedestrian | 1497   | 965       | 0.922 | 0.772 | 0.877 | 0.554 |
| Car        | 1497   | 6500      | 0.945 | 0.96 | 0.982 | 0.873 |

</div>

以`Car`为例说明，精确率`P=0.945`，召回率`R=0.96`，`mAP50=0.982`，`mAP50-95=0.873`。<br>
评估指标显示`Car`的召回率和`mAP50-95`比`Pedestrian`高出较多，这是因为整个`KITTI`数据集中行人的数据量大量少于车的数据量。<br>上述指标的计算公式如[博客](https://blog.csdn.net/qq_63708623/article/details/128508776)所描述。


</div>

## <div align="center">文档</div>

文档前半部分是环境准备，后半部分是如何使用。
### <div align="center">环境准备</div>

<details open>
<summary>安装</summary>

此项目使用的环境是`CUDA-11.8`，`cudnn-8.7.0`，`NVIDIA GeForce GTX 1660 SUPER`。<br>
由于此项目的实验平台是`win11`，安装`C++`版本的`TensorRT`较为复杂，因此选择采用`Python`的`tensorrt`库，该库底层也是通过对`C++`的调用实现，因此可以作为实验对象。
<br>

使用Conda在一个[**Python>=3.8**](https://www.python.org/)环境中安装`tensorrt`包，这里推荐Conda安装，因为`tensorrt`安装与本地环境的`cuda`版本，`cudnn`版本等有关，因此Conda安装成功率较高。

```bash
conda install tensorrt
```
根据本地实验环境的CUDA版本，安装对应的[**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)，下面给出PyTorch2.0.1-cuda11.8的安装命令。
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
安装其余必要的[依赖项](https://github.com/NoMoreBeauty/ultralytics/blob/main/requirements.txt)。

```bash
pip install -r requirements.txt
```
</details>

<details open>
<summary>验证</summary>

在命令行中运行命令：
```bash
python -c "import tensorrt"
```
如果静默则环境安装成功。

</details>

### <div align="center">使用</div>

#### DarkNet_to_onnx
首先将Darknet模型转换为onnx模型，运行：
```bash
python yolov3_to_onnx.py -d .
```
运行成功则在当前目录下会得到yolov3.onnx文件。
<br>

#### onnx_to_trt

再将onnx模型转换为TensorRT适配的trt模型，运行前在`samples/python/yolov3_onnx/`目录下存放一张待检测的照片，例如`dog.jpg`：
```bash
python .\onnx_to_tensorrt.py --img_name dog.jpg -d .
```
运行成功则在当前目录下会得到yolov3.trt文件；注意，该文件同时运行了对图片的检测，因此检测输出的图像在当前目录下`bboxes.png`。
<img width="1024" src="https://github.com/NoMoreBeauty/ultralytics/blob/main/runs/train/results.png" alt="Train Result">
