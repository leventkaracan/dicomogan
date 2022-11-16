# Disentangling Content and Motion for Text-Based Neural Video Manipulation (DicoMoGAN)

- [Introduction](#introduction)
- [Code Requirements](#requirements)
- [Datasets](#data-structure)
    - [Fashion Dataset](#fashion)
    - [3dShapes Dataset](#3dshapes)
- [Training DiCoMoGAN](#training)
- [Testing DiCoMoGAN](#testing)
- [Publications](#publications)

## Introduction

DiCoMoGAN is a video editing framework for manipulating videos with natural language, aiming to perform local and semantic edits on a video clip to alter the appearances of an object of interest. Our GAN architecture allows for better utilization of multiple observations by disentangling content and motion to enable controllable semantic edits. To this end, we introduce two tightly coupled networks: (i) a representation network for constructing a concise understanding of motion dynamics and temporally invariant content, and (ii) a translation network that exploits the extracted latent content representation to actuate the manipulation according to the target description. Our qualitative and quantitative evaluations demonstrate that DiCoMoGAN significantly outperforms existing frame-based methods, producing temporally coherent and semantically more meaningful results.


![](architecture.png)

![](teaser.png)

For more details, you can reach the [paper](https://arxiv.org/abs/2211.02980). 
## Installation

DiCoMOGAN is coded with PyTorch

DiCoMOGAN requires the following installations:

```
python 3.8.3
pytorch (1.7.1)
cuda 11.1
```


## Data Structure

Given a dataset root path in which there are folders containing frames and video list  and corresponding descriptions, you can train your own model.
In the following, we are showing main strucure of dataset folder:

`/dataset_root/<video folders>`

`/dataset_root/train_video.txt`

`/dataset_root/train_video_descriptions.txt`

`/dataset_root/test_video.txt`

`/dataset_root/test_video_descriptions.txt`


## Datasets

### 3dShapes
First, we use the [3D Shapes dataset](https://github.com/deepmind/3d-shapes) which is proposed for learning and
assessing factors of variation from data. This dataset has 480K images of 64 × 64 resolution.
There are 6 ground truth independent latent factors. They are floor color, wall color, object
color, scale, shape and orientation. For our purpose, we build simple text descriptions which
covers object related latent factors object color, scale and shape, e.g. “There is a big blue
capsule.”. To prevent scale ambiguity, we remove two elements of the scale factor which
is of length 8, originally. In that case, “small”, “medium” and “big” in the descriptions
correspond to the first two, middle two and the last two values, respectively. Moreover, we
consider the orientation factor as a dynamic dimension taking 15 different values. We have
19.2K train and 4.8K test videos with 15 frames and simple text descriptions for each video. 

If you use this dataset in your work, please cite related work in [3D Shapes dataset](https://github.com/deepmind/3d-shapes).


### Fashion Video

We collected our dataset from raw videos present in the website of an
online clothing retailer by searching products in the cardigans, dresses, jackets, jeans, jump-
suits, shorts, skirts, tops and trousers categories. Figure given below illustrates these nine garment types
and the variety within each category with some samples. There are 3178 video clips
(approximately 109K distinct frames), which we split into 2579 for training and 598 for
testing.
Garment descriptions. We obtained textual descriptions of the clothes from the headers
of the html files by extracting the hierarchy and info sections of the items. These product
descriptions give details about its color, material properties and design details. We only
manually removed the repetitive or ill-suited words from the descriptions. 

# Please send [us](levent.karacan@iste.edu.tr) a request e-mail to download dataset.

![](fashion.png)


If you use this dataset in your work, please cite our work.

## Training DiCoMoGAN


## Testing DiCoMoGAN*



## Publications

Please cite the following paper if you use DiCoMOGAN and proposed Fashion Dataset.

```
@article{karacan2022disentangling,
  title={Disentangling Content and Motion for Text-Based Neural Video Manipulation},
  author={Karacan, Levent and Kerimo{\u{g}}lu, Tolga and {\.I}nan, {\.I}smail and Birdal, Tolga and Erdem, Erkut and Erdem, Aykut},
  journal={arXiv preprint arXiv:2211.02980},
  year={2022}
}
```
