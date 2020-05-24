# Self-Supervised Learning for Domain Adaptation on Point-Clouds

<p align="center"> 
    <img src="./resources/arch.png" width="400">
</p> 
 
 ### Introduction
Self-supervised learning (SSL) allows to learn useful representations from unlabeled data and has been applied effectively for domain adaptation (DA) on images. It is still unknown if and how it can be
leveraged for domain adaptation for 3D perception. Here we describe the
first study of SSL for DA on point-clouds. We introduce a new pretext
task, Region Reconstruction, motivated by the deformations encountered
in sim-to-real transformation. We also demonstrate how it can be combined with a training procedure motivated by the MixUp method. Evaluations on six domain adaptations across synthetic and real furniture
data, demonstrate large improvement over previous work.

[[Paper]](https://arxiv.org/pdf/2003.12641.pdf)

### Instructions
Clone repo and install it
```bash
git clone https://github.com/idanachi/RegRec_and_PCM.git
cd RegRec_and_PCM
pip install -r requirements.txt
```

Download data:
```bash
cd ./data
python download.py
```

Run PCM on source and RegRec on target
```bash
cd ./scripts
bash regrec_pcm_run.sh
```

Run RegRec on both source and target (without PCM)
```bash
cd ./scripts
bash regrec_run.sh
```


### Citation
Please cite this paper if you want to use it in your work,
```
@article{achituve2020self,
  title={Self-Supervised Learning for Domain Adaptation on Point-Clouds},
  author={Achituve, Idan and Maron, Haggai and Chechik, Gal},
  journal={arXiv preprint arXiv:2003.12641},
  year={2020}
}
```
 
### Region Reconstruction
<p align="center"> 
    <img src="./resources/reconstruction.png">
</p> 
 
 
### Acknowledgement
Some of the code in this repoistory was taken (and modified according to needs) from the follwing sources:
[[PointNet]](https://github.com/charlesq34/pointnet), [[PointNet++]](https://github.com/charlesq34/pointnet2), [[DGCNN]](https://github.com/WangYueFt/dgcnn), [[PointDAN]](https://github.com/canqin001/PointDAN), [[Reconstructing_space]](http://papers.nips.cc/paper/9455-self-supervised-deep-learning-on-point-clouds-by-reconstructing-space), [[Mixup]](https://github.com/facebookresearch/mixup-cifar10)


