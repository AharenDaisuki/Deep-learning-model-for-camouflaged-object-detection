# Deep-learning-model-for-camouflaged-object-detection

This repo is for BSCCS Final Year Project 2023-2024, *Deep-learning model for camouflaged object detection*. [[Final Report](https://drive.google.com/file/d/1hTAkfLB3w7eAXHaP47lbP5EZrpRNHKWT/view?usp=sharing)]

* We propose a novel two-branch VCOD model that utilizes both RGB frames and motion estimations as inputs to give dense predictions of targets in a video.
* Different from other methods using implicit motion handling, we continue to use explicit optical flows in spite of acceptable estimation errors, for the purpose of saving computational cost. To this end, we introduce inter-branch feature fusion module to fuse feature maps obtained from two modalities, i.e., RGB frames and optical flow estimations. Besides, we also introduce intra-branch feature fusion module to fuse feature maps at multiple scales in order to get fine-grained features. 
* Our proposed framework is non-trivial, which achieves 39 FPS (> 20 FPS) inference speed with a competitive performance among state-of-the-art methods on VCOD tasks. Our model outperforms the best method in the evaluation by 9.5% on MAE (mean absolute error) with 55.86 MB (67.82%) less parameters. However, we also observe that the state-of-the-art method outperforms our model on all other metrics. 

![alt text](./img/qualitative_results.png)

## 1. Proposed Model

<p align="left">
    <img src="./img/proposed_framework.png" width='1400' height='300' /> <br />
    <em>
    Figure 1: Illustration of our model. First, the transformer-based backbone extracts
    features from two consecutive input frames and the corresponding motion estimation.
    Then, the proposed Inter-branch Feature Fusion module and Intra-branch Feature Fusion
    module fuse features from both appearance branch and motion branch.
    </em>
</p>

Our model is implemented in [PyTorch](https://github.com/pytorch/pytorch) framework and trained on a single NVIDIA GeForce RTX 3080 GPU of 10240 MiB memory.

## 2. Get Started

**1. Download Datasets**

Download via this [link](https://drive.google.com/file/d/1Y53kXm412YUT9fpVzvirwHDWLTHq_yjp/view?usp=sharing). Highly based on [MoCA-Mask](https://github.com/XuelianCheng/SLT-Net?tab=readme-ov-file), we make some minor modifications in order to make it adapt to the pipeline of our proposed framework: 

* Add Optical Flow Data. For every two consecutive frames, we use [RAFT](https://github.com/princeton-vl/RAFT) to generate their corresponding optical flow data as auxiliary input. The additional optical flow data includes 22852 images in JPG format for 87 videos in total.

* Resize Images with Inconsistent Resolution. To eliminate the effect of inconsistent resolutions, we resize all images to 720 × 1280 as described in our setting.

Your dataset is supposed to be arranged as follows: 

```shell
├── dataset
    ├── flow
    ├── frame
    ├── mask
```

**2. Download Checkpoints**

We adopt MiT-b2 as our backbones. You can download pretrained models via: [MiT-b2](https://drive.google.com/file/d/1ZWpvO8p7DyaVi3lpLLYBfl9fS1ADF63b/view?usp=sharing). Please note that you need to change the file path [here](https://github.com/AharenDaisuki/Deep-learning-model-for-camouflaged-object-detection/blob/main/model/mit.py) in order to correctly load your own pretrained models.  

You can also download our model checkpoints via this [link](https://drive.google.com/file/d/1Yi5Mv2oLHQoz9xcrsk2LYw2Aw20eIvRp/view?usp=sharing). 

**3. Environment**


Before following the below instructions, please make sure you have properly installed Conda.You can download our packed virtual environment via this [link](). 

Go to your conda environments directory, create a new directory for your virtual environment: 

```
mkdir myenv
```

Extract all: 
```
tar -xzf fyp.tar.gz -C myenv
```

Activate your environment: 
```
conda activate myenv
```

## 3. Results

Examples of commands are provided below. You might need to adjust some options, for example, the path of your dataset. 

**Train**
```
python train.py --name "demo" --epoch 100 --batchsize 2 --inputsize 352 --stepsize 50 --dataset "C:\Users\xyli45\Desktop\datasets\MoCA-Mask-Flow" --loss "cons+edge"
```

**Inference**
```
python infer.py --checkpoint "model1.pth" --dataset "C:\Users\xyli45\Desktop\datasets\MoCA-Mask-Flow" --savepath "C:\Users\xyli45\Desktop\model1"
```

**Evaluation**
```
python eval.py --dataset "C:\Users\xyli45\Desktop\datasets\MoCA-Mask-Flow" --pred "C:\Users\xyli45\Desktop\predictions" --gt "C:\Users\xyli45\Desktop\datasets\MoCA-Mask-Flow\mask\val" --savetxt log.txt
```

You can download our pretrained models via this [link](https://drive.google.com/file/d/1Yi5Mv2oLHQoz9xcrsk2LYw2Aw20eIvRp/view?usp=sharing). Or, you can also use our [prediction masks](https://drive.google.com/file/d/1XV_zewiO4kawmvmiFq6q7wfP6knV_gfi/view?usp=sharing) to reproduce the same evaluation results as reported in final report. 

## 4. Acknowledgement

I would like to acknowledge the valuable suggestions and critiques from my final year project supervisor, Professor Rynson W. H. Lau. Also, I would like to thank Dr. Lin Jiaying for his contributions to this project at early stage, who gave me a lot of constructive suggestions about the directions to follow. 