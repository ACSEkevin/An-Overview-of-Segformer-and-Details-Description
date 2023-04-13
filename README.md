# An Overview of Segformer and Details Description
In this repository, the structure of the <b>Segformer</b> model is explained. In many recent blog posts and tutorials, the structure of Segformer has been misunderstood by many people, even experienced computer vision engineers, for reasons that may include misleading diagrams of the Segformer structure in the [original paper](https://arxiv.org/pdf/2105.15203.pdf), but the model structure is shown clearly in the [source code address](https://github.com/NVlabs/SegFormer) given in the paper. Therefore, the details of the Segformer, including <b>OverlapPatchEmbedding</b>, <b>Efficient Multihead Attention</b>, <b>Mixed-FeedForward Network</b>, <b>OverlapPatchMerging</b> and <b>Segformer block</b>, will also be elaborated here. If there is any problem, please feel free to make a complain, also make a [contact](hzhang205@sheffield.ac.uk) if convenient.<br>
Also, the model has been uploaded for a reference which is developed by Keras/TensorFlow.
## Basics and File Description
Project cloning:

```
https://github.com/ACSEkevin/An-Overview-of-Segformer-and-Details-Description.git
```

`ADEChallengeData2016/`: ADE20K,the dataset has been used for training and testing the model, please refer to: [ADE20K Dataset](https://github.com/CSAILVision/ADE20K).<br>
`models/`: Two types of programming the model: <b>structrual</b>  and <b>class inheritance</b>.<br>
`adedataset.py`: a dataset batch generator (keras requirement).<br>
`train.py`: model train script.<br>

## A General Overview of the Model Arcitecture
Here a re-drawn architecture replaces the one from the [original paper](https://arxiv.org/pdf/2105.15203.pdf), which could help with understanding.<p>
<img src="images/seg_former_arch.png" alt="drawing" width="800"/><p>
To conclude and compare:
* In encoder, an input image is scaled to its $\frac{1}{32}$ and then is upsampled to $\frac{1}{4}$ of the original size. However, the model given in the repository upsampled to the full size to attempt for a better result. This can be revised after cloning.

