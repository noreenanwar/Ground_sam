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
