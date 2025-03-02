<h1 align="center"> DAMM: Dual-Stream Attention with Multi-Modal Queries for Unified
Semantics-Geometry Object Detection </h1>

<h2 align="center">

</h2>

This repository contains the implementation of the following paper:
> **DAMM: Dual-Stream Attention with Multi-Modal Queries for Unified
Semantics-Geometry Object Detection**<br>
## Abstract:

Transformer-based object detectors often struggle with occlusions, fine-grained localization, and computational inefficiency caused by fixed queries and dense attention. We propose DAMM, Dual-stream Attention with Multi-Modal queries, a novel framework introducing both query adaptation and structured cross-attention for improved accuracy and efficiency. DAMM capitalizes on three types of queries: appearance-based queries from vision-language models, positional queries using polygonal embeddings, and random queries for broader scene coverage. A \textbf{dual-stream cross-attention} module separately refines semantic and spatial features, boosting localization precision in cluttered scenes. We evaluate DAMM on four challenging benchmarks: Cityscapes, UAVDT, VisDrone, and UA-DETRAC, achieving state-of-the-art performance in average precision (AP) and recall. Our results underscore the effectiveness of multi-modal query adaptation and dual-stream attention in advancing transformer-based object detection.
> 

  
<p align="center">
  <img width=95% src="image.png">
</p>



## Comparison between Baseline and DAMM:
<p align="center">
  <img width=95% src="fig1.png">
</p>





## Installation

### Requirements
- Python >= 3.7, CUDA >= 10.1
- PyTorch >= 1.7.0, torchvision >= 0.6.1
- Cython, COCOAPI, scipy, termcolor

The code is developed using Python 3.8 with PyTorch 1.7.0.
First, clone the repository locally:
```shell
git clone https://github.com/Atten4Vis/ConditionalDETR.git
```
Then, install PyTorch and torchvision:
```shell
conda install pytorch=1.7.0 torchvision=0.6.1 cudatoolkit=10.1 -c pytorch
```
Install other requirements:
```shell
cd DAMM
pip install -r requirements.txt
```



## Usage

### Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
├── annotations/  # annotation json files
└── images/
    ├── train2017/    # train images
    ├── val2017/      # val images
    └── test2017/     # test images
```



### Training

To train DAMM on a single node with 8 gpus for 50 epochs run:
## License

Conditional DETR is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.



## Citation


