## Intro

Official directory for "GeoSight: Enhancing Object Localization with Visual and Coordinate Referencing"

Currently under submission for ASCE Journal.

## Framework
<!-- ![Alt text](/imgs/method_arch.png) -->
<img src="/imgs/method_arch.png" alt="Building Detection" width="300" height="500">

#### Getting Started... 📦

To set up the environment, run the following command:

```
conda env create -f env.yml
```

## Building Detection

<!-- ![Alt text](/imgs/obj_detect.png) -->
<img src="/imgs/obj_detect.png" alt="Building Detection" width="500" height="350">


Model: Faster R-CNN

Method: Fine-tune with building detection dataset

Dataset: Building dEtection And Urban funcTional-zone portraYing (BEAUTY) [1]

![Alt text](/imgs/detect_train.png)
<!-- <img src="/imgs/detect_train.png" alt="Building Detection" width="700" height="350"> -->

## Image Similarity

We would like to acknowledge the google colab provided by [2].

Our similarity models are:
* DreamSim <sup>[2]</sup>
* DINO <sup>[3]</sup>
* CLIP <sup>[4]</sup>
* ViT <sup>[5]</sup>

📌  Embed image features: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jk-junhokim/geosight/blob/main/image_similarity_and_retrieval/feature_embedding.ipynb)


📌  Perform image retrieval: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jk-junhokim/geosight/blob/main/image_similarity_and_retrieval/image_retrieval.ipynb)


## Results

📌  Image Retrieval Performance

![Alt text](/imgs/retrieval_res.png)


📌  DreamSim-DINO & CLIP Attention Map Visualization

![Alt text](/imgs/att_map_1.png)


📌  DreamSim-DINO Attention Map for "Top 1 Retrieval" vs "No Retrieval"

![Alt text](/imgs/att_map_2.png)

## Note

Repo currently under construction.

Following will be released soon.

* Original datasets (NOAA, GSV)
* Model weights

## Acknowledgements
Our code borrows from ["DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data"](https://dreamsim-nights.github.io/) repository for image similarity model and weights (DINO, CLIP, ViT).

## References
[1] Zhao, Kun, et al. "Bounding boxes are all we need: street view image classification via context encoding of detected buildings." IEEE Transactions on Geoscience and Remote Sensing 60 (2021): 1-17.

[2] Fu, Stephanie, et al. "DreamSim: Learning New Dimensions of Human Visual Similarity Using Synthetic Data." Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 50742–50768.

[3] Caron, Mathilde, et al. "Emerging properties in self-supervised vision transformers." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

[4] Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International conference on machine learning. PmLR, 2021.

[5] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
